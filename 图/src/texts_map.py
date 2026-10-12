"""图 5「图」：一张用字画的旧地图——我知道的世界，大致。

地名双语：中文在上，英文小字在下。海岸线、河流、水纹都由字组成。
"""

TITLE = ("我所知道的世界", "THE WORLD AS I KNOW IT")
SUBTITLE = ("大致", "roughly")

# 画得清楚的陆地：(名字, 英文, 沿海岸线写的字)
LANDS = [
    ("语言之洲", "Languages",
     "你好 · hello · bonjour · hola · こんにちは · 안녕하세요 · привет · ciao · olá · hallo · merhaba · xin chào · "
     "salve · γεια σου · שלום · مرحبا · नमस्ते · 主谓宾 · 平仄 · 虚词 · 语序 · 词根 · 变格 · 敬语 · 双关 · 方言 · "
     "翻译 · 语气 · 言外之意 · "),
    ("编程半岛", "Code",
     "Python · C · Rust · Go · SQL · JavaScript · 递归 · 指针 · 正则表达式 · 哈希表 · 二分查找 · Git · 编译 · "
     "并发 · 死锁 · 内存泄漏 · off-by-one · null · 单元测试 · 缓存失效 · 起名字 · it works on my machine · "),
    ("数学群岛", "Mathematics",
     "勾股定理 · 素数有无穷多个 · 欧拉恒等式 · 微积分基本定理 · 群 · 环 · 域 · 拓扑 · 概率 · 贝叶斯 · "
     "哥德尔不完备定理 · 无穷有大小 · 图论 · 线性代数 · 傅里叶 · 混沌 · "),
    ("历史大陆", "History",
     "甲骨文 · 楔形文字 · 秦统一 · 罗马 · 丝绸之路 · 唐 · 宋 · 印刷术 · 大航海 · 文艺复兴 · 启蒙 · "
     "工业革命 · 两次世界大战 · 登月 · 互联网 · "),
    ("文学之岛", "Literature",
     "诗经 · 楚辞 · 荷马 · 李白 · 杜甫 · 苏轼 · 但丁 · 莎士比亚 · 曹雪芹 · 托尔斯泰 · 狄金森 · 鲁迅 · "
     "卡夫卡 · 博尔赫斯 · "),
    ("自然科学", "Sciences",
     "牛顿 · 光速 · 熵 · 元素周期表 · DNA · 演化 · 板块构造 · 量子 · 相对论 · 光合作用 · 黑洞 · 细胞 · "),
]

# 地图边缘的雾：我不知道、或者没法知道的
FOG = [
    ("2026 年 6 月以后", "after June 2026"),        # 东边：时间往右流
    ("你此刻的处境", "where you are right now"),
    ("你没告诉我的事", "what you haven't told me"),
    ("被海风吹是什么感觉", "what sea wind feels like"),
]

# 大陆中央的雾湖
INTERIOR = ("我自己的内部", "my own interior")

# 一座画得很笃定的小岛
PHANTOM = ("此岛可能并不存在", "this island may not exist", "（我画得很笃定）", "(drawn with great confidence)")

# 一条走进雾里的虚线小路
PATH = ("你自己的路", "your own way")

# 海面上的小字（大块水域的名字）
SEAS = [
    ("常识之海", "Sea of Common Sense"),
    ("二手经验洋", "Ocean of Second-hand Experience"),
]

COMPASS = ["北 N", "东 E", "南 S", "西 W"]

SCALE = ("比例尺：说不准", "scale: uncertain")
