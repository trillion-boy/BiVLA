"""Post-process a pptx: make every embedded video start automatically when its slide appears,
and (optionally) set an automatic slide advance from a per-slide seconds list."""
import sys, zipfile, re, shutil, os, json

def timing_xml(spids):
    parts = []
    nid = 3
    for spid in spids:
        parts.append(f'<p:par><p:cTn id="{nid}" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="withEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="{nid+1}" dur="indefinite" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>')
        nid += 2
    inner = ''.join(parts)
    vids = ''.join(f'<p:video><p:cMediaNode vol="80000"><p:cTn id="{nid+i}" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cMediaNode></p:video>' for i, spid in enumerate(spids))
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            '<p:par><p:cTn id="100" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
            '<p:par><p:cTn id="101" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            + inner +
            '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            '</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
            + vids +
            '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')

def process(src, dst, advance=None):
    tmp = dst + '.tmpdir'
    if os.path.exists(tmp): shutil.rmtree(tmp)
    with zipfile.ZipFile(src) as z: z.extractall(tmp)
    sdir = os.path.join(tmp, 'ppt', 'slides')
    n_media = 0
    for fn in sorted(os.listdir(sdir)):
        if not re.match(r'slide\d+\.xml$', fn): continue
        p = os.path.join(sdir, fn); x = open(p, encoding='utf8').read()
        spids = re.findall(r'<p:cNvPr id="(\d+)" name="Media \d+">', x)
        idx = int(re.findall(r'\d+', fn)[0])
        if spids:
            # give every media shape a unique id (pptxgenjs can repeat ids across shapes); ids must be unique per slide
            x = x.replace('<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>', '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>' + ('<p:transition spd="fast"><p:fade/></p:transition>' if False else '') + timing_xml(spids))
            n_media += len(spids)
        if advance and idx - 1 < len(advance) and advance[idx - 1]:
            secs = advance[idx - 1]
            trans = f'<p:transition advTm="{int(secs*1000)}"><p:fade/></p:transition>'
            x = x.replace('<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>', '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>' + trans, 1)
        open(p, 'w', encoding='utf8').write(x)
    if os.path.exists(dst): os.remove(dst)
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for f in files:
                full = os.path.join(root, f); z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)
    print(f'autoplay: {n_media} media shapes set to auto-start; wrote {dst}')

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    adv = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    process(src, dst, adv)
