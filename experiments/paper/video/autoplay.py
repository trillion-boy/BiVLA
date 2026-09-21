"""Post-process a pptx: give every shape a unique id (pptxgenjs reuses ids for media), make every
embedded video start automatically when its slide appears, and set automatic slide advance."""
import sys, zipfile, re, shutil, os, json

def renumber(x):
    """Make every <p:cNvPr id=...> unique within the slide; return (xml, media_ids)."""
    nxt = [2]
    media = []
    def rep(m):
        nxt[0] += 1
        nid = nxt[0]
        if m.group(2).startswith('Media '): media.append(nid)
        return f'<p:cNvPr id="{nid}" name="{m.group(2)}"'
    x = re.sub(r'<p:cNvPr id="(\d+)" name="([^"]*)"', rep, x)
    return x, media

def timing_xml(spids):
    nid = 5
    effects = []
    for spid in spids:
        effects.append(
            f'<p:par><p:cTn id="{nid}" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="withEffect">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr>'
            f'<p:cTn id="{nid+1}" dur="1000" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>')
        nid += 2
    videos = []
    for spid in spids:
        videos.append(f'<p:video><p:cMediaNode vol="80000"><p:cTn id="{nid}" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn>'
                      f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cMediaNode></p:video>')
        nid += 1
    return ('<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            '<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
            '<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            + ''.join(effects) +
            '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            '</p:childTnLst></p:cTn>'
            '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
            '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
            + ''.join(videos) +
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
        x, media = renumber(x)
        idx = int(re.findall(r'\d+', fn)[0])
        x = re.sub(r'<p:transition[^>]*>.*?</p:transition>|<p:transition[^>]*/>', '', x, flags=re.S)
        x = re.sub(r'<p:timing>.*?</p:timing>', '', x, flags=re.S)
        anchor = '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
        extra = ''
        if advance and idx - 1 < len(advance) and advance[idx - 1]:
            # plain cut: a transition effect would add its own duration on top of advTm in the exported video
            extra += f'<p:transition advTm="{int(advance[idx - 1] * 1000)}"/>'
        if media:
            extra += timing_xml(media); n_media += len(media)
        assert anchor in x, fn
        x = x.replace(anchor, anchor + extra, 1)
        open(p, 'w', encoding='utf8').write(x)
    if os.path.exists(dst): os.remove(dst)
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml must be first for strict readers
        ct = os.path.join(tmp, '[Content_Types].xml'); z.write(ct, '[Content_Types].xml')
        for root, _, files in os.walk(tmp):
            for f in files:
                full = os.path.join(root, f); rel = os.path.relpath(full, tmp)
                if rel != '[Content_Types].xml': z.write(full, rel)
    shutil.rmtree(tmp)
    print(f'autoplay: {n_media} media shapes set to auto-start; wrote {dst}')

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    adv = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    process(src, dst, adv)
