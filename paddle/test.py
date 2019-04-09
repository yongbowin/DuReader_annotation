import re

text = "对 文 件 点 右 键 -- 打 开 方 式 -- 选 择 程 序 ， 在 列 表 总 或 浏 览 找 到 对 应 的 " \
       "合 适 程 序 ， 点 选 “ 始 终 使 用 选 择 的 程 序 打 开 这 种 文 件 ”。 可 能 是 误 操 作 " \
       "指 向 了 未 知 的 一 种 打 开 方 式 而 改 变 了 文 件 默 认 的 打 开 方 式 而 变 成 了 未 " \
       "知 图 标 。</p></li><li><p>"

text1 = "就 是 县 城 破 了 点 <br/><br/> 于 是 , 重 点 推 荐 仙 都"


def remove_html(text):
    reg = re.compile(r'<[^>]+>', re.S)
    text = reg.sub('', text)

    return text


print(remove_html(text1))
