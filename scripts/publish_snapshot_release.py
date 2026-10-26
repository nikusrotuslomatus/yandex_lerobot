"""Verify this snapshot on GitHub, then publish its existing draft release.

Invoked by the owner's publication workflow. Never rewrites an existing tag,
uploads data, executes release text, or prints authentication material.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error
import urllib.parse


ROOT=Path(__file__).resolve().parents[1]


def api(path,token,method='GET',payload=None):
    body=None if payload is None else json.dumps(payload).encode()
    request=urllib.request.Request('https://api.github.com/'+path,data=body,method=method,
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                 'X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
        except (urllib.error.URLError,TimeoutError):
            if attempt==2:raise
            time.sleep(2)


def verify_assets(expected,actual):
    by_name={a['name']:a for a in actual}
    if len(by_name)!=len(actual):raise RuntimeError('Duplicate remote asset names')
    if set(by_name)!={a['name'] for a in expected}:raise RuntimeError('Missing or unexpected remote assets')
    for item in expected:
        asset=by_name[item['name']]
        if asset['state']!='uploaded' or asset['size']!=item['bytes']:
            raise RuntimeError('Remote size/state mismatch: '+item['name'])
        if asset.get('digest')!='sha256:'+item['sha256']:
            raise RuntimeError('Remote SHA256 mismatch: '+item['name'])
    return [{'name':x['name'],'bytes':x['size'],'digest':x['digest'],'state':x['state']} for x in actual]


def wait_for_tests(repo,sha,token):
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        data=api(f'repos/{repo}/commits/{sha}/check-runs',token)
        runs=[r for r in data['check_runs'] if r['name']=='pytest']
        if runs:
            latest=max(runs,key=lambda x:x['id'])
            if latest['status']=='completed':
                if latest['conclusion']!='success':raise RuntimeError('Software CI did not pass')
                return {'name':latest['name'],'conclusion':latest['conclusion'],'url':latest['details_url']}
        time.sleep(10)
    raise TimeoutError('Software CI did not finish within 15 minutes')


def save_proof(proof):
    target=ROOT/'results/github_publication/REMOTE_PUBLICATION.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
    subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
    subprocess.run(['git','add','--',str(target.relative_to(ROOT))],check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode
    if changed:
        subprocess.run(['git','commit','-m','Record verified release assets and publication status'],check=True)
        subprocess.run(['git','push','origin','HEAD:main'],check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    expected=json.loads((ROOT/'docs/RELEASE_ASSETS.json').read_text())
    repo=expected['repository'];sha=os.environ['GITHUB_SHA'];token=os.environ['GH_TOKEN']
    if os.environ.get('GITHUB_REPOSITORY')!=repo or os.environ.get('GITHUB_REF')!='refs/heads/main':
        raise RuntimeError('Publication is limited to the configured owner repository main branch')
    proof={'success':False,'repository':repo,'validation_commit':sha,'tag':expected['tag'],
           'workflow_url':f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
           'hardware_acceptance':False}
    try:
        metadata=api('repos/'+repo,token)
        proof['expected_private']=expected['private']
        proof['observed_private']=metadata['private']
        if metadata['private']!=expected['private']:raise RuntimeError('Repository visibility changed')
        release=api(f"repos/{repo}/releases/{expected['release_id']}",token)
        if release['tag_name']!=expected['tag']:raise RuntimeError('Unexpected release tag')
        proof['assets']=verify_assets(expected['assets'],release['assets'])
        proof['software_ci']=wait_for_tests(repo,sha,token)
        if args.publish and release['draft']:
            release=api(f"repos/{repo}/releases/{release['id']}",token,'PATCH',
                {'draft':False,'prerelease':True,'target_commitish':sha,
                 'body':(ROOT/'docs/RELEASE_NOTES.md').read_text(encoding='utf-8')})
        if args.publish:
            if release['draft'] or not release['prerelease']:raise RuntimeError('Release not published as prerelease')
            ref=api(f"repos/{repo}/git/ref/tags/{urllib.parse.quote(expected['tag'],safe='')}",token)['object']
            while ref['type']=='tag':ref=api(f"repos/{repo}/git/tags/{ref['sha']}",token)['object']
            proof.update(published=True,release_url=release['html_url'],snapshot_commit=ref['sha'])
        else:proof['published']=not release['draft']
        proof['success']=True
    except Exception as error:
        proof['error']=type(error).__name__+': '+str(error)
    if args.publish:save_proof(proof)
    print(json.dumps(proof,indent=2))
    if not proof['success']:raise SystemExit(1)


if __name__=='__main__':main()
