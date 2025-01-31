# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
from dataclasses import dataclass

import torch
from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm
from torch import nn
from torch.nn import LayerNorm

import esm as esm
from esm import Alphabet
from esm.esmfold.v1.categorical_mixture import categorical_lddt
from esm.esmfold.v1.trunk import FoldingTrunk, FoldingTrunkConfig
from esm.esmfold.v1.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)
import cryoER.imggen_torch as igt
import numpy as np
def compute_eed(coords):
    """
    Compute the end to end distance for a batch of protein structures.
    """
    first_atom = coords[:, 0, :]
    last_atom = coords[:, -1, :]
    squared_diff = (first_atom - last_atom) ** 2
    squared_distance = torch.sum(squared_diff, dim=1)
    distance = torch.sqrt(squared_distance)
    distance = distance.view(-1, 1)
    return distance

def compute_helix_distances(coords):
    """
    Compute distances between residues i, i+3 and i, i+4 for a batch of coordinates.

    Parameters:
        coords (torch.Tensor): A tensor of shape (N_batch, N_res, 3) containing the coordinates.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors of shape (N_batch, N_res-4) containing
                                           distances between i, i+3 and i, i+4 respectively.
    """
    diff_i3 = coords[:, :-3, :] - coords[:, 3:, :]
    dist_i3 = torch.norm(diff_i3, dim=-1)

    # i to i+4 distances
    diff_i4 = coords[:, :-4, :] - coords[:, 4:, :]
    dist_i4 = torch.norm(diff_i4, dim=-1)

    return torch.cat([dist_i3,dist_i4],dim = 1)

def compute_ca_distances(coords):
    diff_i1 = coords[:, :-1, :] - coords[:, 1:, :]
    dist_i1 = torch.norm(diff_i1, dim=-1)
    return dist_i1
def compute_cryo_image(coord, ref, input_quat=None):
    n_pixel = 128  # number of pixels
    pixel_size = 0.2  # pixel size in Angstrom
    sigma = 1.5  # width of atom in Angstrom
    snr = np.infty  # signal-to-noise ratio
    defocus_min = 0.0579
    defocus_max = 0.058
    rotation = True
    batch_size = 32
    add_ctf = False
    device = coord.device
    R, t = find_alignment_kabsch_batch(coord, ref)
    coord = torch.matmul(R, ref.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    coord = coord - coord.mean(1).unsqueeze(1)
    coord = coord.to(torch.float64)

    images, quats = igt.generate_images_simple(coord,
                                               n_pixel=n_pixel,  ## use power of 2 for CTF purpose
                                               pixel_size=pixel_size,
                                               sigma=sigma,
                                               snr=snr,
                                               add_ctf=add_ctf,
                                               defocus_min=defocus_min,
                                               defocus_max=defocus_max,
                                               batch_size=batch_size,
                                               rotation=rotation,
                                               device=device,
                                               input_quat=input_quat)
    return images.reshape(images.shape[0], -1), quats



def compute_rg(coords):
    """
    Compute the radius of gyration for a batch of protein structures.
    """
    centered_coords = coords - coords.mean(dim=1, keepdim=True)
    sq_distances = torch.sum(centered_coords ** 2, dim=2)
    rg = torch.sqrt(torch.mean(sq_distances, dim=1))
    rg = rg.view(-1, 1)
    return rg




def find_alignment_kabsch_batch(P, Q):
    """Find alignment using Kabsch algorithm between two sets of points P and Q for batches.

    Args:
    P (torch.Tensor): A tensor of shape (B, N, 3) representing the first set of points in batches.
    Q (torch.Tensor): A tensor of shape (B, N, 3) representing the second set of points in batches.

    Returns:
    Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor is the rotation matrix R
    and the second tensor is the translation vector t. The rotation matrix R is a tensor of shape (B, 3, 3)
    representing the optimal rotation for each batch, and the translation vector t is a tensor of shape (B, 3)
    representing the optimal translation for each batch.

    """
    B, N, _ = P.shape
    # Shift points w.r.t centroid
    centroid_P = P.mean(dim=1, keepdim=True)
    centroid_Q = Q.mean(dim=1, keepdim=True)
    P_c, Q_c = P - centroid_P, Q - centroid_Q

    # Find rotation matrix by Kabsch algorithm
    H = torch.matmul(P_c.transpose(1, 2), Q_c)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)

    # ensure right-handedness
    d = torch.sign(torch.linalg.det(torch.matmul(V, U.transpose(1, 2))))

    diag_values = torch.cat([
        torch.ones(B, 1, dtype=P.dtype, device=P.device),
        torch.ones(B, 1, dtype=P.dtype, device=P.device),
        d.unsqueeze(1)
    ], dim=1)

    M = torch.eye(3, dtype=P.dtype, device=P.device).unsqueeze(0).repeat(B, 1, 1)
    M[:, range(3), range(3)] = diag_values

    R = torch.matmul(V, torch.matmul(M, U.transpose(1, 2)))

    # Find translation vectors
    t = centroid_Q - torch.matmul(R, centroid_P.transpose(1, 2)).transpose(1, 2)

    return R, t.squeeze(1)


def calculate_rmsd_batch(pos, ref):
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref for batches.

    Args:
    pos (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the first set of points in batches.
    ref (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the second set of points in batches.

    Returns:
    torch.Tensor: RMSD between the two sets of points for each batch.

    """
    if pos.shape[1] != ref.shape[1]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch_batch(ref, pos)
    ref0 = torch.matmul(R, ref.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    rmsd = torch.linalg.norm(ref0 - pos, dim=2).mean(dim=1).view(-1, 1)
    return rmsd


def compute_COM(batch_tensor):
    com = batch_tensor.mean(dim=1)
    return com


def calculate_disp_batch(pos1, pos2, ref1):
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref for batches.

    Args:
    pos (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the first set of points in batches.
    ref (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the second set of points in batches.

    Returns:
    torch.Tensor: RMSD between the two sets of points for each batch.

    """
    if pos1.shape[1] != ref1.shape[1]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch_batch(pos1, ref1)
    pos1_0 = torch.matmul(R, pos1.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    pos2_0 = torch.matmul(R, pos2.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    pos1_com = compute_COM(pos1_0)
    pos2_com = compute_COM(pos2_0)
    return pos2_com - pos1_com





@dataclass
class ESMFoldConfig:
    trunk: T.Any = FoldingTrunkConfig()
    lddt_head_hid_dim: int = 128


class ESMFold(nn.Module):
    def __init__(self, esmfold_config=None, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg

        self.distogram_bins = 64

        self.esm, self.esm_dict = esm.pretrained.esm2_t36_3B_UR50D()

        self.esm.requires_grad_(False)
        self.esm.half()

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = FoldingTrunk(**cfg.trunk)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(cfg.trunk.structure_module.c_s),
            nn.Linear(cfg.trunk.structure_module.c_s, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(
            self, esmaa: torch.Tensor
    ) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False
        )
        esm_s = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def forward(
            self,
            aa: torch.Tensor,
            mask: T.Optional[torch.Tensor] = None,
            residx: T.Optional[torch.Tensor] = None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            input_structure=None,
            input_fix=None,
            input_fix_mask=None,
            esm_s_input=None,
            operator_list=[0, 0, 0, 0, 0, 0, 0, 0,0],
            operator=[compute_eed, compute_rg, compute_helix_distances, calculate_rmsd_batch, compute_cryo_image,
                      calculate_rmsd_batch, compute_ca_distances, calculate_disp_batch,
                      compute_ca_distances],
            input_quat=None,
            multimer_indice=None,
            multimer_fix=None
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """

        if mask is None:
            mask = torch.ones_like(aa)
        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)
        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)
        esm_s = self._compute_language_model_representations(esmaa)
        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        if esm_s_input is not None:
            esm_s = esm_s + esm_s_input
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        if esm_s.requires_grad == False:
            esm_s.requires_grad = True
        esm_s_output = esm_s.clone().detach().cpu()

        # esm_s = esm_s.detach() #was detach
        # === preprocessing ===
        esm_s_processed = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s_processed)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)
        s_s_0 += self.embedding(aa)
        structure: dict = self.trunk(
            s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
               in [
                   "s_z",
                   "s_s",
                   "frames",
                   "sidechain_frames",
                   "unnormalized_angles",
                   "angles",
                   "positions",
                   "states",
               ]
        }

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        lddt_head = self.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, self.lddt_bins
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = 100 * plddt  # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        ptm_logits = self.ptm_head(structure["s_z"])

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = torch.stack([
            compute_tm(batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins)
            for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
        ])
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self.distogram_bins
            )
        )
        quats = None
        all_output_list = []
        for i in range(len(operator_list)):
            if operator_list[i] == 0:
                all_output_list.append(None)
            else:
                if i == 3:
                    '''
                    RMSD w.r.t a structure, input structure here is the Ca of each residue, so shape N_batch * N_res * 3
                    '''
                    if input_structure is not None:
                        output = operator[3](structure["positions"][-1, :, :, 1, :], input_structure)
                    else:
                        output = None
                elif i == 4:
                    #cryo-EM
                    '''
                    cryo-EM of a structure, input_structure is the Ca of each residue and only used for fixing global
                    rotation so can be any structure but recommended to be the native structure, shape N_batch * N_res * 3
                    input_quat is the quaternion for rotation matrix to be optimized, shape N_batch * 4
                    '''
                    output, quats = operator[4](structure["positions"][-1, :, :, 1, :], input_structure, input_quat)
                elif i == 5:
                    '''
                    CG-to-all-atom backmapping, input_fix represents the CG structure, and in this example an alpha-carbon
                    based CG structure, so of shape N_batch * N_res * 3. We also give freedom if only partial Ca is known,
                    where in this case an input_fix_mask is provided, which is a list of residue index where Ca position
                    is known
                    '''
                    #CG-to-all-atom backmapping
                    if input_fix is not None:
                        if input_fix_mask is None:
                            output = operator[5](structure["positions"][-1, :, :, 1, :], input_fix)
                        else:
                            output = 0
                            tot_len = 0
                            for mask_indice_input in range(len(input_fix_mask)):
                                tot_len += len(input_fix_mask[mask_indice_input])
                                output += operator[5](
                                    structure["positions"][-1, :, input_fix_mask[mask_indice_input], 1, :],
                                    input_fix[mask_indice_input]) * len(input_fix_mask[mask_indice_input])
                            output = output / tot_len
                    else:
                        output = None
                elif i == 7:
                    '''
                    Protein complex alternative binding mode, multimer_fix represents the Ca position of a fixed multimer structure
                    and is only used for fixing global rotation, recommended to be the native structure (to guarantee same linkage
                    between two proteins, the structure is better predicted with ESMFold). Multimer indice is a 2 * 2 array that represents
                    start and end indice of both proteins.
                    '''
                    if multimer_indice is None or multimer_fix is None:
                        output = None
                    else:
                        pos_1 = structure["positions"][-1, :, multimer_indice[0][0]:multimer_indice[0][1], 1, :]
                        pos_2 = structure["positions"][-1, :, multimer_indice[1][0]:multimer_indice[1][1], 1, :]
                        output = calculate_disp_batch(pos_1, pos_2, multimer_fix)
                elif i == 8:
                    '''
                    The optimization can lead to physical breaking of bonds in rare cases. Thus for multimer task it is recommended to set
                    operator_list[8] == 1
                    '''
                    if multimer_indice is None:
                        output = None
                    else:
                        pos_1 = structure["positions"][-1, :, multimer_indice[0][0]:multimer_indice[0][1], 1, :]
                        pos_2 = structure["positions"][-1, :, multimer_indice[1][0]:multimer_indice[1][1], 1, :]
                        ca_dist1 = compute_ca_distances(pos_1)
                        ca_dist2 = compute_ca_distances(pos_2)
                        output = torch.cat([ca_dist1, ca_dist2], dim=-1)

                else:
                    '''
                    The optimization can lead to physical breaking of bonds in rare cases. Thus for monomer task it is recommended to set
                    operator_list[6] == 1
                    '''
                    output = operator[i](structure["positions"][-1, :, :, 1, :])
                all_output_list.append(output)
        return structure, esm_s_output, all_output_list, quats

    # @torch.no_grad()
    def infer(
            self,
            sequences: T.Union[str, T.List[str]],
            residx=None,
            masking_pattern: T.Optional[torch.Tensor] = None,
            num_recycles: T.Optional[int] = None,
            residue_index_offset: T.Optional[int] = 512,
            chain_linker: T.Optional[str] = "G" * 25,
            input_structure=None,
            input_fix=None,
            input_fix_mask=None,
            operator_list=[0, 0, 0, 0, 0,0,1,0,0],
            input_quat=None,
            esm_s_input=None,
            multimer_indice=None,
            multimer_fix=None,
    ):
        """Runs a forward pass given input sequences.

        Args:
            sequences (Union[str, List[str]]): A list of sequences to make predictions for. Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles (cfg.trunk.max_recycles), which is 4.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )
        # print(mask)

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )
        output,esm_s_output, all_output, image_output = self.forward(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
            input_structure=input_structure,
            input_fix=input_fix,
            input_fix_mask=input_fix_mask,
            esm_s_input=esm_s_input,
            input_quat=input_quat,
            operator_list=operator_list,
            multimer_indice=multimer_indice,
            multimer_fix=multimer_fix
        )

        output["atom37_atom_exists"] = output[
                                           "atom37_atom_exists"
                                       ] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output,esm_s_output, all_output, image_output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], esm_s_input=None, *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output, esm_s_output, all_output, image_output = self.infer(
            seqs, esm_s_input=esm_s_input, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        result, rmsd, rmsd_grad, esm_s_output, all_output, image_output, check_embed_token, ground_truth_embeddings = self.infer_pdbs(
            [sequence], *args, **kwargs)
        return result[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
