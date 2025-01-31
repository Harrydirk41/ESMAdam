import torch
import MDAnalysis as mda

def mdau_to_pos_arr(u, frame_cluster=None, select_ca=True,add_num = 0):
    atoms = u.select_atoms("protein and name CA") if select_ca else u.atoms
    if frame_cluster is None:
        n_frame = len(u.trajectory)
    else:
        n_frame = len(frame_cluster)
    pos = torch.zeros((n_frame, len(atoms), 3), dtype=float)
    if frame_cluster is None:
        for i, ts in enumerate(u.trajectory):
            pos[i] = torch.from_numpy(atoms.positions)
    else:
        for i, ts in enumerate(u.trajectory[frame_cluster]):
            pos[i] = torch.from_numpy(atoms.positions)
    pos += add_num
    pos -= pos.mean(1).unsqueeze(1)
    print("pos",pos.shape)
    return pos