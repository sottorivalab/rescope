import argparse
import os
import sys
from pathlib import Path
import json
sys.path.append(os.path.abspath("/home/andrey.tyshevich/GitLab/CellViT-plus-plus/"))
import cellvit
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--cellvit_res',
                        type=str, required=True,
                        help='path with cellvit results')
    return parser.parse_args()


def main():
    args = parse_arguments()
    cellvit_res = Path(args.cellvit_res)
    cells_embeds = torch.load(cellvit_res, weights_only=False)
    torch.save(cells_embeds.x, 'cells_image_embeddings.pt')
    torch.save(cells_embeds.positions, 'cells_image_positions.pt')
    with open('cells_image_metadata.json', 'w') as f:
        json.dump(cells_embeds.metadata, f)


if __name__ == "__main__":
    main()
