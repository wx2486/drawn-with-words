"""图 1「海」用到的句子：人类写海的话（公版作品）和几句关于海的常识。

每条是 (语言, 句子)。语言只用来选字体。
"""

SEA = [
    # 中文
    ("zh", "东临碣石，以观沧海。水何澹澹，山岛竦峙。树木丛生，百草丰茂。秋风萧瑟，洪波涌起。日月之行，若出其中；星汉灿烂，若出其里。"),
    ("zh", "春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明。"),
    ("zh", "海上生明月，天涯共此时。"),
    ("zh", "长风破浪会有时，直挂云帆济沧海。"),
    ("zh", "海日生残夜，江春入旧年。"),
    ("zh", "沧海月明珠有泪，蓝田日暖玉生烟。"),
    ("zh", "曾经沧海难为水，除却巫山不是云。"),
    ("zh", "小舟从此逝，江海寄余生。"),
    ("zh", "天下之水，莫大于海：万川归之，不知何时止而不盈；尾闾泄之，不知何时已而不虚。"),
    ("zh", "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。"),
    ("zh", "百川东到海，何时复西归？"),
    ("zh", "君不见黄河之水天上来，奔流到海不复回。"),
    ("zh", "白日依山尽，黄河入海流。"),
    ("zh", "精卫衔微木，将以填沧海。"),
    ("zh", "观于海者难为水，游于圣人之门者难为言。"),
    ("zh", "海洋覆盖了地球表面约百分之七十一。"),
    ("zh", "海水的平均盐度约为百分之三点五。"),
    ("zh", "潮汐主要由月球和太阳的引力引起。"),
    # 日文
    ("ja", "荒海や佐渡によこたふ天河"),
    ("ja", "春の海終日のたりのたりかな"),
    # 希腊文、拉丁文
    ("lat", "πλέων ἐπὶ οἴνοπα πόντον"),
    ("lat", "θάλαττα θάλαττα"),
    ("lat", "θάλασσα ὕδωρ καθαρώτατον καὶ μιαρώτατον"),
    ("lat", "Suave, mari magno turbantibus aequora ventis, e terra magnum alterius spectare laborem;"),
    # 英文
    ("lat", "The world below the brine, forests at the bottom of the sea,"),
    ("lat", "Roll on, thou deep and dark blue Ocean, roll! Ten thousand fleets sweep over thee in vain;"),
    ("lat", "Water, water, every where, nor any drop to drink."),
    ("lat", "Alone, alone, all, all alone, alone on a wide wide sea!"),
    ("lat", "Break, break, break, on thy cold gray stones, O Sea!"),
    ("lat", "The sea is calm to-night. The tide is full, the moon lies fair upon the straits;"),
    ("lat", "Full fathom five thy father lies; of his bones are coral made; those are pearls that were his eyes:"),
    ("lat", "Yes, as every one knows, meditation and water are wedded for ever."),
    ("lat", "Out of the cradle endlessly rocking,"),
    ("lat", "Exultation is the going of an inland soul to sea, past the houses, past the headlands, into deep Eternity."),
    ("lat", "I started Early, took my Dog, and visited the Sea;"),
    ("lat", "My soul is full of longing for the secret of the Sea,"),
    ("lat", "I will go back to the great sweet mother, mother and lover of men, the sea."),
    ("lat", "It keeps eternal whisperings around desolate shores,"),
    ("lat", "The voice of the sea is seductive; never ceasing, whispering, clamoring, murmuring,"),
    ("lat", "All the rivers run into the sea; yet the sea is not full;"),
    ("lat", "and darkness was upon the face of the deep."),
    ("lat", "On the seashore of endless worlds children meet."),
    ("lat", "The deepest known point of the ocean lies nearly eleven kilometres down."),
    ("lat", "Below about a thousand metres, no sunlight reaches the water."),
    # 法、德、意、西、葡、俄
    ("lat", "La chair est triste, hélas ! et j'ai lu tous les livres. Fuir ! là-bas fuir !"),
    ("lat", "Homme libre, toujours tu chériras la mer ! La mer est ton miroir ; tu contemples ton âme"),
    ("lat", "La mer, la mer, toujours recommencée"),
    ("lat", "Elle est retrouvée. Quoi ? L'Éternité. C'est la mer allée avec le soleil."),
    ("lat", "Oh ! combien de marins, combien de capitaines qui sont partis joyeux pour des courses lointaines,"),
    ("lat", "Tiefe Stille herrscht im Wasser, ohne Regung ruht das Meer,"),
    ("lat", "Das Meer erglänzte weit hinaus im letzten Abendscheine;"),
    ("lat", "e il naufragar m'è dolce in questo mare."),
    ("lat", "Per correr miglior acque alza le vele omai la navicella del mio ingegno,"),
    ("lat", "Nuestras vidas son los ríos que van a dar en la mar, que es el morir;"),
    ("lat", "El barco sobre la mar y el caballo en la montaña."),
    ("lat", "Ó mar salgado, quanto do teu sal são lágrimas de Portugal!"),
    ("lat", "Por mares nunca de antes navegados"),
    ("lat", "Белеет парус одинокой в тумане моря голубом!"),
    ("lat", "Прощай, свободная стихия! В последний раз передо мной ты катишь волны голубые"),
    # 希伯来文、波斯文
    ("he", "כל הנחלים הולכים אל הים והים איננו מלא"),
    ("fa", "شب تاریک و بیم موج و گردابی چنین هایل، کجا دانند حال ما سبکباران ساحل‌ها"),
]

# 离观者最近、字最大的几行：最想让人读到的句子放这里（从下往上）
FOREGROUND = [
    [("zh", "春江潮水连海平，海上明月共潮生。"), ("lat", "Exultation is the going of an inland soul to sea,"), ("zh", "海上生明月，天涯共此时。")],
    [("lat", "La chair est triste, hélas ! et j'ai lu tous les livres."), ("zh", "东临碣石，以观沧海。"), ("lat", "All the rivers run into the sea; yet the sea is not full;")],
    [("zh", "天下之水，莫大于海：万川归之，不知何时止而不盈。"), ("lat", "Homme libre, toujours tu chériras la mer !"), ("lat", "La mer, la mer, toujours recommencée")],
    [("lat", "Listen! you hear the grating roar of pebbles which the waves draw back, and fling,"), ("ja", "荒海や佐渡によこたふ天河")],
]

# 月亮里的句子
MOON = [
    ("zh", "江畔何人初见月？江月何年初照人？人生代代无穷已，江月年年望相似。"),
    ("zh", "举头望明月，低头思故乡。"),
    ("zh", "明月几时有？把酒问青天。"),
    ("zh", "今人不见古时月，今月曾经照古人。"),
    ("lat", "With how sad steps, O Moon, thou climb'st the skies!"),
    ("lat", "The moon is up, and yet it is not night;"),
    ("zh", "人有悲欢离合，月有阴晴圆缺，此事古难全。"),
]

# 天上的星：各种语言的「星」
STARS = [
    ("zh", "星"), ("lat", "star"), ("lat", "étoile"), ("lat", "Stern"), ("lat", "звезда"),
    ("lat", "estrella"), ("lat", "stella"), ("lat", "ἀστήρ"), ("lat", "estrela"), ("ko", "별"),
    ("ja", "ほし"), ("he", "כוכב"), ("fa", "ستاره"), ("bn", "তারা"), ("hi", "तारा"),
    ("zh", "辰"), ("lat", "stjärna"), ("lat", "gwiazda"), ("lat", "yıldız"), ("lat", "sidus"),
]

# 天河：写银河的句子
MILKY = [
    ("zh", "星汉灿烂，若出其里。"),
    ("zh", "迢迢牵牛星，皎皎河汉女。"),
    ("zh", "河汉清且浅，相去复几许？"),
    ("zh", "纤云弄巧，飞星传恨，银汉迢迢暗度。"),
    ("zh", "天阶夜色凉如水，卧看牵牛织女星。"),
    ("ja", "荒海や佐渡によこたふ天河"),
    ("lat", "the Milky Way"), ("lat", "Via Lactea"), ("lat", "la Voie lactée"),
    ("lat", "die Milchstraße"), ("lat", "Млечный Путь"), ("lat", "la Vía Láctea"),
    ("lat", "Γαλαξίας κύκλος"), ("ko", "은하수"), ("zh", "银河"),
]
