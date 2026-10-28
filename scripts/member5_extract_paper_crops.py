"""Extract complete Fig.1/3/4 panels from the supplied five-page IEEE PDF.

Requires Poppler pdftoppm and Pillow. Crops include every axis and legend;
the normalized rectangles below refer specifically to this paper's layout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image

RECTANGLES = {'fig1': (2, (.515,.075,.935,.300)),
              'fig3': (4, (.530,.075,.930,.248)),
              'fig4': (4, (.530,.267,.930,.437))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pdftoppm',default='pdftoppm')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    # Keep rendering in the selected output directory, including sandboxed Windows.
    with tempfile.TemporaryDirectory(dir=args.out.resolve(),prefix='.render-') as temp:
        if not Path(temp).resolve().is_relative_to(args.out.resolve()):
            raise ValueError('Temporary rendering path escaped the output directory')
        for page in (2,4):
            target=Path(temp)/f'page{page}'
            subprocess.run([args.pdftoppm,'-f',str(page),'-l',str(page),'-scale-to','3200',
                            '-singlefile','-png',str(args.pdf),str(target)],check=True)
            with Image.open(target.with_suffix('.png')) as image:
                for name,(number,rectangle) in RECTANGLES.items():
                    if number==page:
                        box=tuple(round(value*(image.width if i%2==0 else image.height)) for i,value in enumerate(rectangle))
                        image.crop(box).save(args.out/f'{name}.png')
    manifest=dict(source_pdf_name=args.pdf.name,source_pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                  source_pages_1_based_and_normalized_rectangles=RECTANGLES,render_page_max_dimension=3200,
                  extraction_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.png')})
    (args.out/'source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
