"""Check output pairs, embedded PDF fonts, and render all PDFs with Poppler."""
from pathlib import Path
import json,subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
from common import OUT
specs=json.loads((OUT/'figure_manifest.json').read_text())
qa=OUT/'qa';qa.mkdir(exist_ok=True)
def check(s):
    name=s['name'];pdf=OUT/'pdf'/(name+'.pdf');png=OUT/'png'/(name+'.png')
    assert pdf.exists() and png.exists() and (OUT/'scripts'/('plot_'+name+'.py')).exists()
    with Image.open(png) as im:
        im.verify()
    with Image.open(png) as im:
        assert abs(im.info['dpi'][0]-600)<1
    fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
    assert 'TimesNewRoman' in fonts and 'Type 3' not in fonts,fonts
    for line in fonts.splitlines()[2:]:
        assert 'yes' in line,line
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    assert 'Pages:           1' in info
    subprocess.run(['pdftoppm','-scale-to','1200','-singlefile','-png',str(pdf),str(qa/name)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    return {'figure':name,'fonts':fonts,'single_page':True,'png_dpi':600}
with ThreadPoolExecutor(max_workers=4) as pool:reports=list(pool.map(check,specs))
(qa/'verification.json').write_text(json.dumps(reports,indent=2))
for start in range(0,len(specs),9):
    batch=specs[start:start+9];sheet=Image.new('RGB',(1800,350*((len(batch)+2)//3)),'#e5e5e5');draw=ImageDraw.Draw(sheet)
    for i,s in enumerate(batch):
        im=Image.open(qa/(s['name']+'.png')).convert('RGB');im.thumbnail((590,316));x=(i%3)*600;y=(i//3)*350
        sheet.paste(im,(x+(600-im.width)//2,y+27));draw.text((x+5,y+6),s['name'],fill='black')
    sheet.save(qa/f'overview_{start//9+1}.png')
print(f'Verified and PDF-rendered all {len(reports)} figures.')
