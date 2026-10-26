"""Lossless release packaging of all generated project work, with checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT=Path(__file__).resolve().parents[1]


def package(output):
    output=Path(output).resolve()
    for directory in ('results','.planning_library','presentation_beamer'):
        source=ROOT/directory
        if output==source or source in output.parents:raise ValueError('Archive output intersects its input')
    output.mkdir(parents=True,exist_ok=False)
    inventory=[];archives=[]
    for directory,name in (('results','experiment-results.zip'),('.planning_library','planning-library.zip'),
                           ('presentation_beamer','presentation-working-files.zip')):
        source=ROOT/directory
        paths=sorted(p for p in source.rglob('*') if p.is_file())
        if any(output==p.parent or output in p.parents for p in paths):raise ValueError('Archive output intersects its input')
        target=output/name;total=0
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
            for path in paths:
                if path.is_symlink():raise ValueError(f'Refusing symlink: {path}')
                relative=path.relative_to(ROOT).as_posix();digest=hashlib.sha256();size=0
                info=zipfile.ZipInfo.from_file(path,arcname=relative);info.compress_type=zipfile.ZIP_DEFLATED
                with path.open('rb') as src,archive.open(info,'w',force_zip64=True) as dest:
                    while chunk:=src.read(1024*1024):dest.write(chunk);digest.update(chunk);size+=len(chunk)
                if size!=path.stat().st_size:raise RuntimeError(f'Input changed while packaging: {path}')
                inventory.append({'archive':name,'path':relative,'bytes':size,'sha256':digest.hexdigest()});total+=size
        with zipfile.ZipFile(target) as archive:
            bad=archive.testzip()
            if bad is not None:raise RuntimeError(f'ZIP CRC failed: {bad}')
        with target.open('rb') as stream:archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        archives.append({'name':name,'bytes':target.stat().st_size,'sha256':archive_hash,
                         'entries':len(paths),'uncompressed_bytes':total,'crc_valid':True})
        print(f'{name}: {len(paths)} files, {target.stat().st_size/1024**2:.1f} MiB; CRC verified',flush=True)
    detailed=output/'artifact-inventory.json'
    detailed.write_text(json.dumps({'schema':1,'files':inventory},indent=2),encoding='utf-8')
    archives.append({'name':detailed.name,'bytes':detailed.stat().st_size,
                     'sha256':hashlib.sha256(detailed.read_bytes()).hexdigest(),'entries':len(inventory)})
    manifest={'schema':1,'release':'v0.1.0-research-snapshot-2026-10-26',
              'repository':'nikusrotuslomatus/yandex_lerobot','archives':archives,
              'omitted_regenerable_directories':['.venv','__pycache__','.pytest_cache','third_party/menagerie_source'],
              'hardware_acceptance':False}
    (output/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    package(args.out)
