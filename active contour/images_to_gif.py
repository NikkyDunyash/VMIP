from importlib.metadata import metadata
import os
import sys
import imageio


path=os.path.abspath(sys.argv[1])
filenames=os.listdir(path)
for filename in filenames:
    if os.path.splitext(filename)[1]!='.jpg':
        filenames.remove(filename)
with imageio.get_writer(os.path.join(path, 'ac.gif'), mode='I', fps=2) as writer:
    for filename in sorted(filenames, key=lambda string: int(os.path.splitext(string)[0])):
        image = imageio.imread(os.path.join(path,filename))
        writer.append_data(image)