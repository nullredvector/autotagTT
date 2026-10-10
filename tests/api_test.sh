#!/bin/bash
# Integration test of /api/autotag* against a scratch archive server. usage: api_test.sh <port> <archive_dir>
P=$1; A=$2; B=http://localhost:$P; fail=0
ID1=7000000000000000009; ID2=7000000000000000010; ID3=7000000000000000011
chk() { if [ "$2" == "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got $2 want $3"; fail=1; fi; }
post() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/feedback" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
STARS_BEFORE=$(shasum $A/data/.appdata/stars.json)
chk "GET lists 14 pending" "$(curl -s "$B/api/autotag?tag=braces" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["pending"]))')" 14
chk "GET bad tag -> 400" "$(curl -s -o /dev/null -w '%{http_code}' "$B/api/autotag?tag=%3Cb%3E")" 400
chk "GET unknown tag -> empty" "$(curl -s "$B/api/autotag?tag=zzz" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["pending"]))')" 0
chk "reject ok" "$(post '{"tag":"braces","id":"'$ID1'","decision":"rejected"}')" 200
chk "reject repeat idempotent" "$(post '{"tag":"braces","id":"'$ID1'","decision":"rejected"}')" 200
chk "flip without change -> 409" "$(post '{"tag":"braces","id":"'$ID1'","decision":"accepted"}')" 409
chk "restore with change" "$(post '{"tag":"braces","id":"'$ID1'","decision":"pending","change":true}')" 200
chk "accept ok" "$(post '{"tag":"braces","id":"'$ID2'","decision":"accepted"}')" 200
chk "undo accepted refused" "$(post '{"tag":"braces","id":"'$ID2'","decision":"rejected","change":true}')" 409
chk "unknown id -> 404" "$(post '{"tag":"braces","id":"123","decision":"accepted"}')" 404
chk "malformed id -> 400" "$(post '{"tag":"braces","id":"../../x","decision":"accepted"}')" 400
chk "bad decision -> 400" "$(post '{"tag":"braces","id":"'$ID3'","decision":"delete"}')" 400
chk "bad tag -> 400" "$(post '{"tag":"<x>","id":"'$ID3'","decision":"rejected"}')" 400
chk "invalid JSON -> 400" "$(post '{nope')" 400
chk "array body -> 400" "$(post '[1]')" 400
chk "non-JSON content type -> 415" "$(curl -s -o /dev/null -w '%{http_code}' -X POST $B/api/autotag/feedback -H 'Content-Type: text/plain' -d '{}')" 415
chk "cross-origin -> 403" "$(post '{"tag":"braces","id":"'$ID3'","decision":"rejected"}' -H 'Origin: http://evil.example')" 403
chk "same-origin ok" "$(post '{"tag":"braces","id":"'$ID3'","decision":"rejected"}' -H "Origin: $B")" 200
chk "oversize body -> 413" "$(post "$(python3 -c 'print("{\"x\":\""+"a"*5000+"\"}")')")" 413
chk "GET on feedback -> 405" "$(curl -s -o /dev/null -w '%{http_code}' $B/api/autotag/feedback)" 405
chk "PUT on feedback -> 405" "$(curl -s -o /dev/null -w '%{http_code}' -X PUT $B/api/autotag/feedback)" 405
chk "unsure ok" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure"}')" 200
chk "unsure->accepted needs change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"accepted"}')" 409
chk "unsure->rejected with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"rejected","change":true}')" 200
chk "rejected->unsure with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure","change":true}')" 200
chk "unsure->pending with change" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"pending","change":true}')" 200
chk "unsure again (left set aside)" "$(post '{"tag":"braces","id":"7000000000000000012","decision":"unsure"}')" 200
chk "pending now 12" "$(curl -s "$B/api/autotag?tag=braces" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["pending"]),len(d["accepted"]),len(d["rejected"]));assert len(d["unsure"])==1')" "11 1 1"
chk "feedback file mode 600" "$(stat -f %Lp $A/data/.appdata/autotag_feedback.json 2>/dev/null || stat -c %a $A/data/.appdata/autotag_feedback.json)" 600
chk "server never touched stars.json" "$(shasum $A/data/.appdata/stars.json)" "$STARS_BEFORE"
# ── AI tags + tag picker ──
ai() { curl -s "$B/api/autotag/ai" | python3 -c "$1"; }
chk "AI endpoint: has videos" "$(ai 'import sys,json;d=json.load(sys.stdin);print(d["available"],len(d["videos"])>0)')" "True True"
chk "AI hides video already in same-name human group" "$(ai 'import sys,json;d=json.load(sys.stdin);print("7000000000000000005" in d["videos"])')" False
chk "AI hides decided (rejected id3 braces)" "$(ai 'import sys,json;d=json.load(sys.stdin);print([t for t,_ in d["videos"].get("'$ID3'",[])])')" "['scenery']"
chk "AI-only tag feedback accepted (scenery on AI video)" "$(post '{"tag":"scenery","id":"7000000000000000012","decision":"rejected"}')" 200
chk "AI feedback on a tag it was not tagged with -> 404" "$(post '{"tag":"scenery","id":"7000000000000000040","decision":"rejected"}')" 404
chk "tags GET defaults to braces" "$(curl -s $B/api/autotag/tags)" '{"enabled":["braces"]}'
tp() { curl -s -o /tmp/at_body -w '%{http_code}' -X POST "$B/api/autotag/tags" -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
chk "tags set (case-insensitive, deduped)" "$(tp '{"enabled":["Braces","SCENERY","scenery"]}')" 200
chk "tags saved" "$(curl -s $B/api/autotag/tags)" '{"enabled":["braces","scenery"]}'
chk "tags unknown group -> 404" "$(tp '{"enabled":["no such group"]}')" 404
chk "tags non-array -> 400" "$(tp '{"enabled":"braces"}')" 400
chk "tags bad entry -> 400" "$(tp '{"enabled":[5]}')" 400
chk "tags cross-origin -> 403" "$(tp '{"enabled":["braces"]}' -H 'Origin: http://evil.example')" 403
chk "tags config mode 600" "$(stat -f %Lp $A/data/.appdata/autotag_tags.json 2>/dev/null || stat -c %a $A/data/.appdata/autotag_tags.json)" 600
chk "stars.json still untouched" "$(shasum $A/data/.appdata/stars.json)" "$STARS_BEFORE"
exit $fail
