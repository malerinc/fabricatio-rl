import io
import glob

from nbstripout import strip_output
from nbformat import read, write, NO_CONVERT

# print(dir(nbstripout))


if __name__ == '__main__':
    directory = "./"
    pathname = directory + "/**/*.ipynb"
    notebook_names = glob.glob(pathname, recursive=True)
    
    for notebook in notebook_names:
        print(notebook)
        with io.open(notebook, 'r', encoding='utf8') as f:
            nb = read(f, as_version=NO_CONVERT)
        ek = ['metadata.language_info.version','metadata.language_info.pygments_lexer']
        #for e in ek:
        nb = strip_output(nb, False, False, extra_keys=ek)
        nb.metadata.kernelspec.display_name = "Python 3"
        with io.open(notebook, 'w', encoding='utf8') as f:
            write(nb, f)