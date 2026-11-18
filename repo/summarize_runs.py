
import json, os, glob, pandas as pd
R = "/content/drive/MyDrive/capstone_ai/runs"
rows = []
for d in sorted(glob.glob(f"{R}/*/")):
    n = os.path.basename(d.rstrip("/"))
    try:
        c = json.load(open(f"{d}config.json")); m = pd.read_csv(f"{d}metrics.csv")
    except Exception as e:
        print("bo qua", n, e); continue
    b = m.loc[m.miou.idxmax()]; l = m.iloc[-1]
    bp = os.path.exists(f"{d}best.pt")
    rows.append(dict(run=n, arch=c["arch"], encoder=c["encoder"], dec=c.get("dec", ""), size=f'{c["w"]}x{c["h"]}',
                     epochs_done=int(l.epoch), epochs_cfg=c["epochs"], best_epoch=int(b.epoch),
                     best_miou=round(b.miou, 4), last_miou=round(l.miou, 4),
                     last_iou_floor=round(l.iou_floor, 4), last_false_floor=round(l.false_floor, 4),
                     sec_per_epoch=int(m.sec.iloc[2:].median()) if len(m) > 3 else int(m.sec.median()),
                     best_pt=bp, last_pt=os.path.exists(f"{d}last.pt"),
                     best_mb=round(os.path.getsize(f"{d}best.pt") / 1e6, 1) if bp else 0))
df = pd.DataFrame(rows)
df.to_csv(f"{R}/summary.csv", index=False)
print(df.to_string(index=False))
