import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.imu_dataset import build_dataloader
DATAPATH = "data"
file_list = [
    f"{DATAPATH}/clap_once/clap_once_20251108_191951.hdf5",
    f"{DATAPATH}/clap_once/clap_once_20251108_200621.hdf5"
]

dataloader = build_dataloader(
    file_list,
    batch_size=8,
    target_seq_len=1440,
    imu_normalizer=None,
)

for batch in dataloader:
    print(batch["imu"].shape)         
    print(batch["file"])
    break
