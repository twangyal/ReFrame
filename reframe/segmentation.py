"""Hosted Meta SAM 3.1 integration; official parser, immutable completed cache.

Network calls are opt-in and have not been exercised with a real API credential.
No local SAM model is substituted. See docs/SAM_API.md for the verified contract.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format

MODEL = 'sam-3.1'
OPTIONS = {'mask_encoding': 'one_bit', 'include_confidence': 'true', 'propagation_direction': 'both'}


def request_body(file_id, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('a noun phrase or documented object prompt is required')
    return {'model': MODEL, 'input': [{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_text', 'text': prompt}, {'type': 'input_video', 'file_id': file_id}]}],
        'stream': False, 'metadata': dict(OPTIONS)}


def cache_key(video_sha256, prompt):
    # Raw inference cache is shared by all selected object IDs in this response.
    value = {'schema': 1, 'video_sha256': video_sha256, 'model': MODEL, 'prompt': prompt,
             'options': OPTIONS}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_cached_response(path):
    result = json.loads(Path(path).read_text())
    if result.get('status') != 'completed':
        raise ValueError('cached response is incomplete; it cannot be reused as success')
    return result


def validate_import(payload,video_sha256,manifest_sha256,prompt):
    """Require an export envelope; never stamp unrelated raw text with new provenance."""
    expected={'video_sha256':video_sha256,'frame_manifest_sha256':manifest_sha256,
              'model':MODEL,'prompt':prompt}
    if not isinstance(payload,dict) or any(payload.get('provenance',{}).get(k)!=v for k,v in expected.items()):
        raise ValueError('import requires matching original video/manifest/model/prompt provenance')
    if not isinstance(payload.get('response'),dict):
        raise ValueError('import envelope has no response')
    return payload['response']


def decode_response(response, object_id, n_frames, height, width):
    """Return full-source-resolution masks, observed-target flags and confidence.

    Omitted frames are legal on a COMPLETED response. They are represented by
    zero masks and observed=False and are excluded from association evidence.
    A missing or incomplete response is an error, not an all-background clip.
    The official parser converts inclusive API boxes to half-open mask bounds.
    """
    if response.get('status') != 'completed':
        raise ValueError('SAM response must be completed')
    if min(n_frames, height, width) < 1:
        raise ValueError('invalid source shape')
    parts = []
    for item in response.get('output', []):
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'refusal':
                    raise ValueError('SAM refused the request')
                if part.get('type') == 'output_text':
                    parts.append(part['text'])
    if len(parts) > 1:
        raise ValueError('expected a single segmentation output-text lane')
    text = ''.join(parts)
    # Source dimensions are present in every API box but not retained on the
    # parser's normalized box records. Check them before parsing masks.
    for token in re.findall(r'<\|box;([^|]*)\|>',text):
        try:
            fields=dict(part.split('=',1) for part in token.split(';'))
            dimensions=(int(fields['w']),int(fields['h']))
        except (ValueError,KeyError):
            raise ValueError('invalid source dimensions on SAM box') from None
        if dimensions!=(width,height):
            raise ValueError('SAM source resolution does not match the frame manifest')
    parser = video_segmentation_format().create_parser()
    parser.push(text)
    parsed = parser.finish(CompletedOutcome()).result
    if parsed.diagnostics or any(r.kind == 'text' for r in parsed.records):
        raise ValueError('unrecognized or malformed SAM output; raw response retained')
    masks = np.zeros((n_frames, height, width), dtype=bool)
    present = np.zeros(n_frames, dtype=bool)
    confidence = np.full(n_frames,np.nan,dtype=np.float32)
    seen = set()
    for r in parsed.records:
        if r.frame is None or not 0 <= r.frame.frame_index < n_frames:
            raise ValueError('SAM frame index is outside the manifest')
        if r.kind != 'mask':
            continue
        t = r.frame.frame_index
        identity = (t, r.object_id)
        if identity in seen:
            raise ValueError('duplicate object/frame mask; inspect raw output')
        seen.add(identity)
        if r.object_id != str(object_id):
            continue
        b = r.bounds
        x0,y0,x1,y1 = (int(v) for v in (b.left,b.top,b.right,b.bottom))
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('SAM mask bounds are outside the source frame')
        raw = np.frombuffer(decode_mask_to_raster(r.mask), dtype=np.uint8).reshape(r.mask.height,r.mask.width)
        if raw.shape != (y1-y0,x1-x0):
            raw = np.asarray(Image.fromarray(raw).resize((x1-x0,y1-y0), Image.Resampling.NEAREST))
        masks[t,y0:y1,x0:x1] = raw != 0
        present[t] = True
        if r.confidence is not None:
            confidence[t]=r.confidence
    return masks,present,confidence


def fetch_or_cache(video, video_sha256, prompt, cache_dir, *, allow_api=False):
    """Return completed raw response and cache ID. No retries of billable calls.

    The caller must verify video_sha256 against the file before entering here.
    Requires explicit operator authorization via allow_api on a cache miss.
    """
    from .media import sha256_file
    video = Path(video)
    if sha256_file(video) != video_sha256:
        raise ValueError('video changed since the frame manifest was created')
    key = cache_key(video_sha256,prompt)
    directory = Path(cache_dir)/key
    completed = directory/'response.json'
    if completed.exists():
        return load_cached_response(completed),key
    if not allow_api:
        raise RuntimeError('no completed cache; authorize a billable API call explicitly')
    if not os.environ.get('MODEL_API_KEY'):
        raise RuntimeError('MODEL_API_KEY is missing; set it securely in environment settings')
    from openai import OpenAI
    import httpx,ssl
    # System CA bundle retains managed-environment proxy trust; never disable TLS.
    client = OpenAI(base_url='https://api.meta.ai/v1', api_key=os.environ['MODEL_API_KEY'],
                    max_retries=0, timeout=600.,
                    http_client=httpx.Client(verify=ssl.create_default_context(),timeout=600.))
    directory.mkdir(parents=True,exist_ok=True)
    # Lock prevents two workers from issuing the same paid request concurrently.
    lock = directory/'inflight.lock'
    try:
        lock.open('x').close()
    except FileExistsError:
        raise RuntimeError('a prior request is in flight or its outcome is uncertain; inspect before retrying') from None
    try:
        if completed.exists():
            return load_cached_response(completed),key
        upload = directory/'upload.json'
        if upload.exists():
            file_id=json.loads(upload.read_text())['id']
        else:
            with video.open('rb') as f:
                uploaded=client.files.create(file=f,purpose='user_data')
            file_id=uploaded.id
            upload.write_text(json.dumps({'id':file_id}))
        result=client.responses.create(**request_body(file_id,prompt)).model_dump(mode='json')
        if result.get('status') != 'completed':
            (directory/'incomplete.json').write_text(json.dumps(result))
            raise RuntimeError('SAM did not complete; response retained, no automatic retry')
        temp=directory/'response.tmp'
        temp.write_text(json.dumps(result))
        temp.replace(completed)
        lock.unlink()
        return result,key
    finally:
        # On failure retain the lock: timeout may have completed remotely.
        client.close()
