# Git 协作与最终提交

每位成员使用独立分支，例如 `member/np0`、`member/np1`、`member/baselines`。模块接入遵守 docs/算法接口.md。算法参数和源码必须随结果一起提交；大规模可再生成数据、临时渲染和压缩包忽略。

常用流程（在仓库根目录运行，分支名按本人修改）：

```powershell
git switch -c member/np0
git add team/np0.py algorithms.json
git commit -m "feat: integrate member NP0 implementation"
```

组长合并前检查接口、单元测试和独立进程复跑记录，确认另一人已填写验收表。正式完整实验从确定的提交运行；manifest 保存 commit 与是否含未提交修改。结果文档后续提交时引用实验 commit。

远程地址由小组确定后配置：`git remote add origin <小组仓库地址>`，再按团队权限推送。不要提交密码/token；不要改写他人历史。当前交付只维护本地版本，未替团队创建远程仓库或上传课程平台。

最终打包前：核对全员验收表、完整扫描结果、报告/PPT/讲稿是否齐全，再运行 `python main.py package`。本次框架包尚不能替代这些尚未交付的材料。

