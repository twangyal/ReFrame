# Verified hosted SAM 3.1 contract

Official documentation and one real hosted request were verified on the exact Balloon2 frames.

- [Overview](https://dev.meta.ai/docs/sam/overview)
- [Prompts, limits and billing](https://dev.meta.ai/docs/sam/segmenting)
- [Output grammar and identity caveats](https://dev.meta.ai/docs/sam/reading-segmentation)
- [Official parsers](https://dev.meta.ai/docs/sam/client-libraries)
- [File upload](https://dev.meta.ai/docs/api-reference/files/upload-file)
- [Pricing](https://dev.meta.ai/docs/pricing-rate-limits)
- [Model catalog endpoint](https://dev.meta.ai/docs/api-reference/models/list-models)

Base URL `https://api.meta.ai/v1`; Bearer authentication; docs use the variable
`MODEL_API_KEY`. `POST /files` multipart uses `purpose=user_data`, then
`POST /responses`, `model="sam-3.1"`, an `input_text` prompt plus
`{"type":"input_video","file_id":...}`. Nonstreaming is officially supported
and used by this short-clip prototype. OpenAI Python SDK is configured with
Meta's base URL, preserved system CA trust, and zero automatic retries.

Metadata values are strings: `mask_encoding=one_bit`,
`include_confidence=true`, `propagation_direction=both`. The latter produces
forward then reverse frames for point/box prompts; frame line order is not time
order. Up to 15,000 frames and 16 tracked objects per frame are documented.
The official pricing page, retrieved 2026-10-26, lists video segmentation at
**$0.20 per 1,000 frames**, with a minimum of **50 billed frames per request**.
The 12-frame sample would therefore cost **$0.01** at the published rate.
`usage.video_frames_processed` reports billed frames; retain it with actual
latency/cost after inference. One request completed with
`usage.video_frames_processed=50`, consistent with **$0.01** at this rate.
No additional inference request was issued; this is not invoice verification.

Short noun phrase or point/box prompts, not both together. Example of object ID
1 at the source's actual dimensions (replace coordinates/dimensions):

```
<0f>1<|point;x=160;y=120;w=320;h=240|>
<0f>1<|box;x1=100;y1=80;x2=200;y2=160;w=320;h=240|>
```

One `output_text` lane holds special-token records: `<Nf>` frame marker,
integer object ID, box token, and encoded mask token. No-match frames emit no
line; an entirely empty completed response is valid. IDs can be sparse and are
only guaranteed within the response while objects remain in frame. Meta warns
against assuming persistence across full occlusion, exit/re-entry or long clips.

Use official `meta-sam-parser==0.0.6`, not a handwritten mask codec. The installed
Python package exposes `video_segmentation_format().create_parser()`,
`push(text)`, `finish(CompletedOutcome())`, and `decode_mask_to_raster(record.mask)`.
This differs in naming from the TypeScript examples on the overview page.
Source API boxes have inclusive bottom/right endpoints; the Python parser emits
half-open `record.bounds`. Masks are box-local, not full-frame. It is wrong to
stretch the box mask across the entire image or to treat mask origin `(0,0)` as
the source-image origin. Encoding is custom base85, not base64/ordinary ASCII85.

Tests decode an independently generated synthetic rectangle payload, verifying placement
in its 320×334 source frame, sparse ID 2, reverse order and absent frames. This is
an offline fixture. Separately, the real request’s `one_bit` output decoded with
no parser diagnostics: object0, twelve 480×270 masks, frame indices0–11. All
12 overlays were visually inspected and exact source-frame alignment passed.

Raw response cache identity includes exact video hash, model, prompt and options.
Decoded masks additionally bind the selected ID and exact frame manifest hash.
Incomplete responses, malformed text and resolution mismatches fail. On a
completed response, omitted target frames get zero masks **and observed=False**;
association skips them rather than claiming reliable background evidence.
Confidence is retained separately from object presence. Missing confidence
and scores below `--min-confidence` (default 0.5) are excluded from association
and evaluation; the threshold is an explicit heuristic, not a calibrated guarantee.
Manual overlay review is still required. Imports require a provenance envelope
matching original video digest, exact manifest digest, model and prompt.

The local SAM 3.1 Python predictor is a separately verified alternative, **not
the implemented hosted provider**. No weights were downloaded or gated model
agreements accepted. The parser and model are under Meta's custom SAM License.


## First real request

Prompt **green dinosaur balloon**,
completed in **13.578 seconds including upload and response retrieval**. The
provider reported 50 billed video frames and zero token usage. Original RGB
H.264 input was accepted without re-encoding, cropping or resizing. Input and
frame-manifest hashes bind the cached response, decoded masks and association.

Object ID0 appeared in all 12 frames with confidence0.9336099624633789 repeated
across frames. Treat this as an object score, not twelve independent calibrated
quality estimates. The balloon remains covered during rotation with no broad
spill onto the person/wall in the inspected overlays. Precise boundary error,
full occlusion/re-entry and unseen-view geometry are not established by this
short sequence. No masks were manually edited or replaced with fixtures.

Raw response/cache, provenance, masks and overlays remain local under
`.cache/sam/` and `outputs/balloon2-sam/`; no upstream or derived visual assets
were published. [Numerical evidence](evidence/sam-3.1-2026-10-27.json) is tracked.
Cache reuse for decoding and identity selection does not call the API.
Live inference requires explicit opt-in and provider billing authorization.
