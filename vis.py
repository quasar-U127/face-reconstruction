#!/usr/bin/env python
import argparse

import numpy as np
import visvis as vv


def VisualizeVolume(filename):
    vol = np.load(filename)
    vol = np.swapaxes(vol, 0, 2)
    volRGB = vol
    vv.volshow(volRGB, renderStyle='iso')
    vv.use().Run()


if (__name__ == "__main__"):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--volume", type=str,
                           help="< model weights file >", required=True)
    args = argParser.parse_args()
    VisualizeVolume(args.volume)
