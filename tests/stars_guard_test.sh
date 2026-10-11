#!/bin/bash
# Tests the stale-save guard on /api/stars (compare-and-swap revisions, history snapshots, visible write errors).
# usage: stars_guard_test.sh <port> <archive_dir>   -- run on a FRESH scratch archive: it rewrites stars.json.
P=$1; A=$2; B=http://localhost:$P; fail=0; F=$A/data/.appdata/stars.json; H=$A/data/.appdata/stars-history
chk() { if [ "$2" == "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got $2 want $3"; fail=1; fi; }
rev() { curl -s $B/api/stars | python3 -c 'import sys,json;print(json.load(sys.stdin)["rev"])'; }
put() { curl -s -o /tmp/sg_body -w '%{http_code}' -X PUT $B/api/stars -H 'Content-Type: application/json' "${@:2}" -d "$1"; }
body() { python3 - "$1" "$2" <<'PY'
import sys,json
n=int(sys.argv[1]); rev=sys.argv[2]
d={"stars":{},"groups":[{"id":"g1","name":"Braces","videoIds":[str(7000000000000000000+i) for i in range(n)]}],"levels":{}}
if rev!="-": d["rev"]=rev
print(json.dumps(d))
PY
}
R0=$(rev)
chk "GET returns a 16-char revision" "$(echo -n $R0 | wc -c | tr -d ' ')" 16
SHA0=$(shasum $F)
chk "PUT without revision -> 428" "$(put "$(body 5 -)")" 428
chk "428 leaves the file untouched" "$(shasum $F)" "$SHA0"
chk "PUT with a wrong revision -> 409" "$(put "$(body 5 deadbeefdeadbeef)")" 409
chk "409 leaves the file untouched" "$(shasum $F)" "$SHA0"
chk "409 body gives the current revision, no paths" "$(python3 -c "import json;d=json.load(open('/tmp/sg_body'));print(d['rev']==('$R0'), d['error'], '/' in json.dumps(d))")" "True stale False"
chk "PUT with the current revision -> 200" "$(put "$(body 60 $R0)")" 200
R1=$(python3 -c "import json;print(json.load(open('/tmp/sg_body'))['rev'])")
chk "response revision matches the next GET" "$(rev)" "$R1"
chk "a second writer still holding the old revision is refused" "$(put "$(body 7 $R0)")" 409
chk "the refused writer changed nothing (60 members)" "$(python3 -c "import json;print(len(json.load(open('$F'))['groups'][0]['videoIds']))")" 60
chk "invalid JSON -> 400" "$(put '{nope')" 400
chk "malformed body -> 400 or 428, never a crash" "$(put '[1]' | grep -c -E '^(400|428)$')" 1
# a save that removes >10% of all memberships first snapshots the old file
chk "shrinking save accepted (revision ok)" "$(put "$(body 20 $R1)")" 200
chk "a -shrink snapshot was kept" "$(ls $H 2>/dev/null | grep -c -- '-shrink.json')" 1
chk "history dir 700" "$(stat -f %Lp $H 2>/dev/null || stat -c %a $H)" 700
SNAP=$(ls $H | grep -- '-shrink.json' | head -1)
chk "snapshot holds the old 60 members" "$(python3 -c "import json;print(len(json.load(open('$H/$SNAP'))['groups'][0]['videoIds']))")" 60
chk "snapshot mode 600" "$(stat -f %Lp $H/$SNAP 2>/dev/null || stat -c %a $H/$SNAP)" 600
# an unwritable data folder must be reported, not silently ignored
R2=$(rev)
chmod 555 $A/data/.appdata
CODE=$(put "$(body 20 $R2)")
chmod 755 $A/data/.appdata
chk "unwritable folder -> 500 (visible failure)" "$CODE" 500
chk "500 body has no paths" "$(grep -c '/' /tmp/sg_body)" 0
chk "after permissions are fixed the same save works" "$(put "$(body 20 $R2)")" 200
exit $fail
