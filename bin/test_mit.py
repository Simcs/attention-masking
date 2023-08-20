from ood_with_vit.datasets.mit import MiT
import pickle

# with open(f'./data/mit/metadata/mit_val_metadata.pkl', 'rb') as f:
#     metadata = pickle.load(f)
        
dataset = MiT(
    root='/home/simc/workspace/dataset/Moments_in_Time_Raw',
    annotation_path='/home/simc/workspace/datset/Moments_in_Time_Raw/validationSet.csv',
    frames_per_clip=16,
    step_between_clips=1,
    frame_rate=8,
    split='val',
    num_workers=8,
    # _precomputed_metadata=metadata,
)

print(dataset[0])

with open('/home/simc/workspace/OOD-with-ViT/data/mit/metadata/mit_val_metadata.pkl', 'wb') as f:
    pickle.dump(dataset.metadata, f)

