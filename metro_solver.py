"""
上海地铁谜题求解器 / Shanghai Metro Puzzle Solver
仓库名"213"对应上海地铁1号线、2号线、3号线

本脚本构建了上海地铁网络图，可以：
1. 查询任意两站之间的最短路径
2. 查询换乘站
3. 分析各线路交叉点
"""

from collections import defaultdict, deque

# 上海地铁各线路站点（部分主要站点）
METRO_LINES = {
    1: ["莘庄", "外环路", "莲花路", "锦江乐园", "春申路", "上海南站", "漕宝路",
        "上海体育馆", "徐家汇", "衡山路", "常熟路", "陕西南路", "淮海中路",
        "黄陂南路", "人民广场", "新闸路", "汉中路", "上海火车站", "中山北路",
        "延长路", "上海马戏城", "彭浦新村", "通河新村", "呼兰路", "共富新村",
        "宝安公路", "友谊西路", "富锦路"],

    2: ["徐泾东", "虹桥火车站", "虹桥2号航站楼", "淞虹路", "北新泾", "威宁路",
        "娄山关路", "中山公园", "江苏路", "静安寺", "南京西路", "人民广场",
        "南京东路", "陆家嘴", "东昌路", "世纪大道", "上海科技馆", "世纪公园",
        "龙阳路", "张江高科", "金科路", "唐镇", "创新中路", "华夏东路", "川沙",
        "迪士尼", "浦东国际机场", "广兰路"],

    3: ["江杨北路", "宝杨路", "水产路", "淞滨路", "张华浜", "淞发路",
        "长途客运总站", "上海火车站", "中山北路", "镇坪路", "曹杨路",
        "中山公园", "金沙江路", "延安西路", "虹桥路", "宜山路", "漕溪路",
        "上海体育馆", "上海体育场", "龙华", "龙漕路", "石龙路", "上海南站"],
}

# 一次性构建网络图，供所有函数复用
_GRAPH = None
_STATION_LINES = None


def _get_metro_graph():
    """懒加载：构建并缓存地铁网络图"""
    global _GRAPH, _STATION_LINES
    if _GRAPH is not None:
        return _GRAPH, _STATION_LINES

    graph = defaultdict(dict)  # graph[station] = {neighbor: (line, cost)}
    station_lines = defaultdict(set)  # station_lines[station] = {line numbers}

    for line_num, stations in METRO_LINES.items():
        for station in stations:
            station_lines[station].add(line_num)
        for i in range(len(stations) - 1):
            s1, s2 = stations[i], stations[i + 1]
            graph[s1][s2] = (line_num, 1)
            graph[s2][s1] = (line_num, 1)

    _GRAPH = graph
    _STATION_LINES = station_lines
    return graph, station_lines


def find_transfer_stations(line1, line2):
    """找到两条线路的换乘站"""
    _, station_lines = _get_metro_graph()
    return [s for s, lines in station_lines.items() if line1 in lines and line2 in lines]


def bfs_shortest_path(start, end):
    """BFS求最短路径"""
    graph, _ = _get_metro_graph()
    if start not in graph or end not in graph:
        return None, []

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        if current == end:
            return len(path) - 1, path
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None, []


def analyze_213():
    """分析上海地铁213线路关系"""
    print("=" * 50)
    print("上海地铁 2-1-3 号线换乘站分析")
    print("=" * 50)

    for l1, l2 in [(1, 2), (2, 3), (1, 3)]:
        transfers = find_transfer_stations(l1, l2)
        print(f"\n{l1}号线 ∩ {l2}号线 换乘站: {', '.join(transfers)}")

    all_three = find_transfer_stations_multi([1, 2, 3])
    print(f"\n1号线 ∩ 2号线 ∩ 3号线: {'、'.join(all_three) if all_three else '无直接三线换乘站'}")

    print("\n" + "=" * 50)
    print("关键换乘路径示例")
    print("=" * 50)

    # 示例：人民广场到中山公园
    dist, path = bfs_shortest_path("人民广场", "中山公园")
    if path:
        print(f"\n人民广场 → 中山公园: {' → '.join(path)} ({dist}站)")

    # 示例：上海南站到世纪大道
    dist, path = bfs_shortest_path("上海南站", "世纪大道")
    if path:
        print(f"\n上海南站 → 世纪大道: {' → '.join(path)} ({dist}站)")


def find_transfer_stations_multi(lines):
    """找到多条线路的公共换乘站"""
    _, station_lines = _get_metro_graph()
    return [s for s, slines in station_lines.items() if all(line in slines for line in lines)]


if __name__ == "__main__":
    analyze_213()

    print("\n" + "=" * 50)
    print("谜题答案推断")
    print("=" * 50)
    print("\n基于'213'（上海地铁2、1、3号线）的核心换乘站：")
    print("  2号线 × 1号线 = 人民广场")
    print("  2号线 × 3号线 = 中山公园")
    print("  1号线 × 3号线 = 上海体育馆 / 上海火车站 / 中山北路 / 上海南站")
    print("\n最终答案候选：人民广场 或 中山公园")
