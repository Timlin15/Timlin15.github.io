在配置好obsidian的git插件后，我想能不能直接将笔记库的文件直接转化为可读的博客页面，并且全程要用GitHub Actions自动配置，不用我打一个指令。毕竟前面几次博客都是这么荒废的。
并且这个博客最好原生支持obsidian特殊的语法，比如[[]]内联其他文件，以及文件夹嵌套文件夹的文本储存方式。
简单地用大语言模型检索，就可以发现[Quartz](https://quartz.jzhao.xyz/)是完美符合需求的一个产品。

> [!NOTE] 注意
> 如遇任何问题最好查阅[官方doc](https://quartz.jzhao.xyz/)

## Quartz配置
Quartz的配置非常简单，可以直接照抄Quartz的初始页：
```bash
git clone https://github.com/jackyzha0/quartz.git
cd quartz
npm i
npx quartz create
```
进入选择content文件夹中要有什么内容，由于我的文本主要在另外一个文件夹处，所以我选择了Systemlink，直接输入obsidian笔记的绝对路径，即可把obsidian的笔记库链接到quartz的content文件夹。

> [!NOTE] 注意
> 由于Quartz需要content文件夹中有一个`index.md`的文件，而用systemlink的方式创建content文件夹并不会自动创建这个文件，因此需自行创建。并且可以用两个`---`框住`title:...`来创建标题

然后在`quartz.config.ts`中修改`pageTitle=标题`、`locale=zh-CN`、`baseUrl=username.github.io`这几个属性，用这个命令直接查看预览：
```bash
npx quartz build --serve
```
修改完毕后即可进行下一步：利用GitHub Actions自动同步

## 利用GitHub Actions自动同步
这步的有两个GitHub Actions需要设置：
1. 当在obsidian中更新文件并自动同步到GitHub repo后，触发GitHub Actions使quartz的仓库同步更新。这步是选做，但是如果你不做就无法做到无缝同步；
2. quartz的特性是只要你content文件夹更新并push到GitHub中后，可以[利用GitHub Actions直接更新静态页面](https://quartz.jzhao.xyz/hosting)，不需要自行更新静态页面。这步是必须的。

先从第一步讲起：为了实现两个仓库的联动，需要前往 [Developer Settings](https://github.com/settings/apps)中`Personal access tokens->Fine-grained tokens`创建一个token。需要给出名字，过期日期，Repository access（此处推荐只有你的博客这个repo），并给出code, repository advisories, 和workflows的读写权利，metadata的读权利应该是自行给予的。
有了这个token后，前往你的obsidian的文本GitHub repo，去到`Settings->secrets and variables->Actions`中创建一个secret，填入token，比如此处我取名叫`QUARTZ_PAT`
然后在本地文件处创建一个`.github/workflows/sync.yml`
```yml
name: Sync to Quartz Blog
on:
  push:
    branches: [main]
    paths-ignore:
      - '个人/**'

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout vault
        uses: actions/checkout@v4
        with:
          path: vault

      - name: Checkout quartz
        uses: actions/checkout@v4
        with:
          repository: username/username.github.io
          token: ${{ secrets.QUARTZ_PAT }}
          path: quartz

      - name: Sync content
        run: |
          rm -rf quartz/content
          mkdir -p quartz/content
          rsync -av \
            --exclude '私密/' \
            --exclude '.obsidian/' \
            --exclude '.git/' \
            --exclude '.gitignore' \
            vault/ quartz/content/

      - name: Push to Quartz repo
        run: |
          cd quartz
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git diff --staged --quiet || git commit -m "sync: update blog content $(date -u +%Y-%m-%d)"
          git push
```
其中如果你的token取名不一样，应该修改其中的token名字，同时需要修改其中的repo号，修改为你自己的仓库地址。
以及如果你有不想同步的文件，可以用path-ignore或者.gitignore忽略这个文件。

然后参照官方的hosting教程，仿照上一步创建`.github/workflows/deploy.yml`
```yml
name: Deploy Quartz site to GitHub Pages
 
on:
  push:
    branches:
      - main
 
permissions:
  contents: read
  pages: write
  id-token: write
 
concurrency:
  group: "pages"
  cancel-in-progress: false
 
jobs:
  build:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Fetch all history for git info
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - name: Install Dependencies
        run: npm ci
      - name: Build Quartz
        run: npx quartz build
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: public
 
  deploy:
    needs: build
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```
注意修改branches号，一般都是main。完成这步后设置好settings中的pages，即可全自动同步笔记至博客。