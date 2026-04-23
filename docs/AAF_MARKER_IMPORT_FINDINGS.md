# AAF Marker Import Findings

Date: 2026-04-21

This note captures the FinalPass AAF marker import/export findings from the
interactive probe session against Pro Tools. The goal was to determine whether
AAF can carry both:

- a short marker name/title
- a separate longer marker comment/body

## Executive Summary

- Pro Tools can import AAF markers authored by FinalPass.
- Pro Tools does not appear to export its own markers back out to AAF.
- For imported markers, Pro Tools uses `UserComments["Comment"]` as the visible
  marker name.
- Pro Tools synthesizes the visible comment/details field from:
  - `CommentMarkerUser`
  - the described timeline slot name
  - `UserComments["Comment"]`
- No separately imported long comment field was found in the tested set.
- The practical conclusion is that, through this AAF import path, Pro Tools does
  not appear to support a distinct imported long marker comment in the way we
  wanted.

## Pro Tools Export Check

User-supplied file:

- `resources/session-with-known-marker-export.aaf`

Known source session contents before export:

- Track `Audio 1` comment: `THIS TRACK IS EMPTY`
- Marker 1 name: `Marker Name - Keep Short`
- Marker 1 comment:
  `This is the comments field. This is allowed to be considerably longer, and should be the area where you choose to place the longer information for each event. Once you figure out what this is, DO NOT FORGET!`
- AAF global comment:
  `this-is-the-AAF-comment-field`

Observed in the exported AAF:

- one `CompositionMob`
- one audio timeline slot: `Audio 1`
- one timecode slot
- no `EventMobSlot`
- no `CommentMarker`
- no `DescriptiveMarker`
- no matching text payload for the known marker strings
- top-level mob comment-like tagged values existed but were blank

Conclusion:

- Pro Tools appears to strip markers and related comment payload on AAF export,
  even though it can import markers from external AAFs.

## Probe 1

Goal:

- put a unique token in every plausible marker text field and see what Pro Tools
  displays

Observed result:

- Marker Name:
  - `M1_UC_COMMENT`
  - `M2_UC_COMMENT`
  - blank when `UserComments["Comment"]` was absent
- Marker Comment:
  - `M1_USER - Marker Guide - M1_UC_COMMENT`
  - `M2_USER - Marker Guide - M2_UC_COMMENT`
  - ` - Marker Guide - ` when both user and short-name source were absent

Conclusion:

- visible marker name comes from `UserComments["Comment"]`
- visible marker comment is synthesized as:
  - `CommentMarkerUser + " - " + described slot name + " - " + UserComments["Comment"]`
- the following did not drive visible PT name/comment in this probe:
  - `Comment`
  - `CommentMarkerAnnotationList`
  - `UserComments["Detail"]`
  - `CommentMarkerAttributeList["_ATN_CRM_COM"]`

## Probe 2

Goal:

- compare `CommentMarker` vs `DescriptiveMarker`
- vary one long-comment candidate field at a time while keeping
  `UserComments["Comment"]` fixed

Observed result:

- `CommentMarker` and `DescriptiveMarker` behaved the same for visible Pro Tools
  name/comment display
- `DescribedSlots` did not unlock a separate imported long comment field
- keeping `UserComments["Comment"]` fixed always kept the visible marker name
  fixed
- none of these overrode the visible PT comment/details field:
  - `Comment`
  - `CommentMarkerAnnotationList`
  - `UserComments["Detail"]`
  - `CommentMarkerAttributeList["_ATN_CRM_COM"]`

Conclusion:

- marker class choice did not solve the separate-long-comment problem

## Probe 3

Goal:

- test fallback name sources when `UserComments["Comment"]` is intentionally
  absent
- test whether any other field can override the synthesized comment while
  `UserComments["Comment"]` is present

Observed result:

- No fallback marker-name source was found.
- If `UserComments["Comment"]` was absent, Marker Name stayed blank.
- The following did not populate Marker Name on their own:
  - `Comment`
  - `CommentMarkerAnnotationList`
  - `UserComments["Label"]`
  - `UserComments["Detail"]`
  - `CommentMarkerAttributeList["Comment"]`
  - `CommentMarkerAttributeList["Label"]`
  - `CommentMarkerAttributeList["_ATN_CRM_COM"]`
- The visible Comments column remained synthesized:
  - no user, no short name -> ` - COMMENTPROBE - `
  - user only -> `USER - COMMENTPROBE - `
  - short name only -> ` - COMMENTPROBE - SHORT`
  - user + short name -> `USER - COMMENTPROBE - SHORT`

Important clarification:

- the middle term in the synthesized comment is the described timeline slot
  name, not the visible Track Name column in the marker list

## Final Mapping

Based on the probes, the current best-known Pro Tools AAF import mapping is:

- Marker Name:
  - `UserComments["Comment"]`

- Marker Comment / Details:
  - synthesized by Pro Tools from:
    - `CommentMarkerUser`
    - described timeline slot name
    - `UserComments["Comment"]`

## Fields Tested And Not Found To Provide A Separate Imported Long Comment

- `Comment`
- `CommentMarkerAnnotationList`
- `UserComments["Detail"]`
- `UserComments["Label"]`
- `CommentMarkerAttributeList["Comment"]`
- `CommentMarkerAttributeList["Label"]`
- `CommentMarkerAttributeList["_ATN_CRM_COM"]`

## Practical Guidance For FinalPass

- If FinalPass needs a readable short marker title in Pro Tools, write it to
  `UserComments["Comment"]`.
- If FinalPass wants the Pro Tools Comments column to be less noisy, control:
  - `CommentMarkerUser`
  - the described slot name
- Do not assume AAF can carry a separate long imported marker comment that Pro
  Tools will actually display through this path.
- If future work revisits this, start from this note before running new probe
  matrices.
