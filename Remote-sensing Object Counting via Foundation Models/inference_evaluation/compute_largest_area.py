from pathlib import Path
from statistics import mean, median

def parse_line(line: str, line_no: int):
    """
    parseonerow: name x1 y1 x2 y2
    return (name, x1, y1, x2, y2, area)
    parsefailthenreturn None
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"No. {line_no} rowformatError:shouldfor 5 column, actual {len(parts)} column -> {line}")

    name = parts[0]
    try:
        x1, y1, x2, y2 = map(float, parts[1:])
    except ValueError:
        raise ValueError(f"No. {line_no} rowcoordinatenotisnumbercharacter -> {line}")

    # area: abs guaranteecoordinateorderreversereversealsocanjustconfirmcompute
    area = abs(x2 - x1) * abs(y2 - y1)
    return (name, x1, y1, x2, y2, area)


def analyze_areas_from_file(txt_path: str, topk: int = 10):
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"filenotsavein: {txt_path}")

    records = []
    with txt_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                rec = parse_line(line, i)
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                # encountertobadrowthenskip, andprintprompt
                print(f"[skip] {e}")

    if not records:
        print("noreadtohaseffectdata. ")
        return

    areas = [r[5] for r in records]

    # statistic
    avg_area = mean(areas)
    med_area = median(areas)

    # TopK
    records.sort(key=lambda x: x[5], reverse=True)
    top = records[:topk]

    print(f"\nfile: {txt_path}")
    print(f"haseffectrownumber: {len(records)}")
    print(f"averagearea (mean): {avg_area:.6f}")
    print(f"medianarea (median): {med_area:.6f}")

    print(f"\nareamaximum  {min(topk, len(top))} : \n")
    for rank, (name, x1, y1, x2, y2, area) in enumerate(top, start=1):
        print(f"{rank:02d}. area={area:.6f}  |  {name}  ({x1}, {y1}, {x2}, {y2})")

    print("\npureareanumerical value(TopK, fromlargetosmall): ")
    print([r[5] for r in top])


if __name__ == "__main__":
    # TODO: takeherechangebecomeyou textfilepath
    path = r"dataset/local pseudo/rsoc-building_grid16_train.txt"
    analyze_areas_from_file(path, topk=10)
