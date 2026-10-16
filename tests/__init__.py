"""测试包标记（使 `python -m unittest tests.test_pure_rag` 在任意环境都能定位到本仓库的 tests）

没有 __init__.py 时 tests 只是命名空间包，若环境中存在同名的已安装包
（例如 site-packages 下的 tests 目录），常规包的优先级更高，导入会被它遮挡。
"""
