# Proposed frozen release assets

These files are intentionally not part of the normal Git tree. If distributed, attach them to a versioned GitHub Release and verify SHA-256 before use.

| Release filename | Seed/key | Size | SHA-256 | Scientific role |
|---|---:|---:|---|---|
| `seed_84_best.pt` | 84 | 456,813 bytes | `9297cf5639add3f178b8e8d4f6d5d77fdfde88780411ec7fdbc419973a446991` | Frozen OLD-only contrastive R2 encoder, seed 84. |
| `seed_184_best.pt` | 184 | 456,813 bytes | `8a37f05b7e37fbc54794298c758628120c94ddb1f0b0e6b981459b95c93a695a` | Frozen OLD-only contrastive R2 encoder, seed 184. |
| `seed_284_best.pt` | 284 | 456,813 bytes | `607f45918226755191e6c4f2ba94b2544c38b3d04432e4a7646f28978d776132` | Frozen OLD-only contrastive R2 encoder, seed 284. |
| `round94_memory_embeddings.npz` | `aggregate` and seed embeddings | 656,769 bytes | `8af255ad4e0af8e859b06c58e34c798f2b50988c488e081e610a0f8ed695e607` | Frozen OLD TRAIN embedding bank used for cosine kNN-20. |

The checkpoint hashes match `metadata/round94_model_freeze.json`. The memory is the frozen OLD TRAIN embedding bank and is needed by the inference entry point. No checkpoint or memory archive was copied into the public package or uploaded.
