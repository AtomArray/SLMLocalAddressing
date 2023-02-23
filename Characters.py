import os
import sys
pathDir = os.path.dirname(sys.argv[0])
import numpy as np

CHARACTER_HEIGHT = 6
CHARACTER_WIDTH = 5

characters = []
maps = []

with open(os.path.join(pathDir, 'character_bitmaps.txt')) as bitmap:

    i = 0
    fline = bitmap.readline()
    
    while fline:
        if i % (CHARACTER_HEIGHT + 1) == 0:
            characters.append((fline.strip())[0])
            charArray = []
        else:
            line = fline.replace(' ', '').strip('\n')
            lineArray = np.array(list(line)).astype(int)
            charArray.append(lineArray)
            if i % (CHARACTER_HEIGHT + 1) == CHARACTER_HEIGHT:
                maps.append(np.array(charArray))
            
        # print("Line %i: %s"%(i, line))
        fline = bitmap.readline()
        i += 1

characterDict = dict(zip(characters, maps))

