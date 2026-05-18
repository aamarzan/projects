import sys
buf=[]
w=0
def flush_record(rec):
    global w
    # Basic FASTQ structure checks
    if len(rec)!=4: return
    if not rec[0].startswith("@"): return
    if not rec[2].startswith("+"): return
    if len(rec[1].strip())==0 or len(rec[3].strip())==0: return
    sys.stdout.write("".join(rec))
    w += 1

for line in sys.stdin:
    buf.append(line)
    if len(buf)==4:
        flush_record(buf)
        buf=[]
# drop partial tail silently
