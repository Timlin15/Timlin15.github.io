> 本文件大部分由Claude Opus 4.5生成
## 命令概览

|类别|命令|一句话说明|
|---|---|---|
|**网络**|`curl`|万能 HTTP 客户端，下载/上传/调试接口|
|**网络**|`wget`|下载文件，支持断点续传和递归下载|
|**网络**|`ssh`|远程登录服务器、端口转发、跳板机|
|**网络**|`scp`|通过 SSH 在本地和远程之间拷贝文件|
|**网络**|`rsync`|增量同步文件，比 scp 更智能更快|
|**网络**|`ss`|查看网络连接和端口监听状态|
|**文本**|`grep`|在文件/输出中搜索匹配的文本行|
|**文本**|`sed`|流式文本编辑，批量查找替换|
|**文本**|`awk`|按列处理文本，轻量级数据分析|
|**文本**|`cat`|查看文件内容、拼接文件|
|**文本**|`head` / `tail`|查看文件头部/尾部，实时追踪日志|
|**文本**|`sort` / `uniq`|排序和去重|
|**文本**|`wc`|统计行数、单词数、字符数|
|**文本**|`cut`|按分隔符或位置截取列|
|**文本**|`tee`|同时输出到屏幕和文件|
|**文件**|`find`|按名称/大小/时间等条件查找文件|
|**文件**|`xargs`|将标准输入转为命令参数，批量执行|
|**文件**|`tar`|打包/解包归档文件|
|**文件**|`ln`|创建硬链接或软链接|
|**文件**|`chmod` / `chown`|修改文件权限/所有者|
|**文件**|`df` / `du`|查看磁盘空间 / 目录大小|
|**进程**|`ps`|查看当前进程快照|
|**进程**|`top` / `htop`|实时系统资源监控|
|**进程**|`kill` / `killall`|发送信号终止进程|
|**进程**|`nohup`|让进程在退出终端后继续运行|
|**进程**|`jobs` / `bg` / `fg`|管理后台/前台任务|
|**系统**|`systemctl`|管理 systemd 服务|
|**系统**|`journalctl`|查看 systemd 日志|
|**系统**|`env` / `export`|查看/设置环境变量|
|**系统**|`alias`|给命令起别名|
|**系统**|`lsblk` / `mount`|查看/挂载块设备|
|**其他**|`tmux`|终端复用器，断线不丢会话|
|**其他**|`watch`|定时重复执行命令并刷新输出|
|**其他**|`jq`|命令行 JSON 解析器|
|**其他**|`strace`|跟踪进程的系统调用|
## 符号符
**`|` — 管道符（pipe）**，这是最核心的那个。它把左边命令的输出，作为右边命令的输入：

```bash
ps aux | grep python    # ps 的输出 → 传给 grep 过滤
cat log.txt | sort | uniq -c   # 可以链式连接多个
```
**`||` — 逻辑或**，左边命令失败了才执行右边，**`&&` — 逻辑与**，左边成功了才执行右边。
**`\` — 续行符**，放在行末表示"这一行还没写完，接着下一行"。最终会拼成一条完整命令：
```bash
docker run \ 
	--name mycontainer \ 
	-p 8080:80 \ 
	-v /data:/data \ 
	nginx
```

## Ghostty适配
部分应用不认识Ghostty这个终端，需要export一个其他适配的终端。
```bash
export TERM=xterm-256color
```

## 详细解析

### curl

> 发送 HTTP 请求，调试 API，下载文件

```bash
# 发送 GET 请求
curl https://api.example.com/data

# 发送 POST JSON 请求
curl -X POST -H "Content-Type: application/json" -d '{"key": "value"}' https://api.example.com

# 带 Bearer Token 认证
curl -H "Authorization: Bearer <token>" https://api.example.com

# 下载文件并保存为原始文件名
curl -O https://example.com/file.tar.gz

# 下载文件并指定文件名
curl -o output.txt https://example.com/data

# 跟随重定向（-L）并显示响应头（-i）
curl -Li https://example.com

# 只查看响应头
curl -I https://example.com

# 静默模式 + 只输出 HTTP 状态码
curl -s -o /dev/null -w "%{http_code}" https://example.com

# 通过代理发送请求
curl -x http://proxy:8080 https://example.com
```

---

### wget

> 下载文件，支持断点续传和递归

```bash
# 下载文件
wget https://example.com/file.tar.gz

# 断点续传（下载中断后继续）
wget -c https://example.com/large-file.iso

# 后台下载
wget -b https://example.com/large-file.iso

# 递归下载整个网站（深度 2 层）
wget -r -l 2 https://example.com

# 指定输出文件名
wget -O custom-name.tar.gz https://example.com/file.tar.gz
```

---

### ssh

> 远程登录、端口转发、跳板机

```bash
# 登录远程服务器
ssh user@hostname

# 指定端口和密钥
ssh -p 2222 -i ~/.ssh/mykey user@hostname

# 本地端口转发（把远程 8080 映射到本地 8080）
ssh -L 8080:localhost:8080 user@remote-server

# 远程端口转发（把本地 3000 暴露到远程 9000）
ssh -R 9000:localhost:3000 user@remote-server

# 跳板机连接（通过 bastion 访问 internal）
ssh -J user@bastion user@internal-server

# 执行远程命令后退出
ssh user@hostname "nvidia-smi"

# 保持连接不断（配合 ~/.ssh/config 更佳）
ssh -o ServerAliveInterval=60 user@hostname
```

---

### scp

> 通过 SSH 拷贝文件

```bash
# 上传本地文件到远程
scp local-file.txt user@remote:/path/to/dest/

# 下载远程文件到本地
scp user@remote:/path/to/file.txt ./

# 递归拷贝目录
scp -r ./local-dir user@remote:/path/to/dest/

# 指定端口
scp -P 2222 file.txt user@remote:/path/
```

---

### rsync

> 增量同步，比 scp 更快更智能

```bash
# 同步本地目录到远程（注意尾部 / 的区别）
rsync -avz ./src/ user@remote:/dest/

# 从远程同步到本地
rsync -avz user@remote:/src/ ./dest/

# 同步时删除目标端多余的文件
rsync -avz --delete ./src/ user@remote:/dest/

# 显示进度
rsync -avz --progress ./src/ ./dest/

# 排除特定文件/目录
rsync -avz --exclude='node_modules' --exclude='.git' ./project/ user@remote:/project/

# 模拟运行（不实际执行，只看会做什么）
rsync -avzn ./src/ ./dest/
```

---

### ss

> 查看网络连接和端口监听（替代 netstat）

```bash
# 查看所有监听端口
ss -tlnp

# 查看所有 TCP 连接
ss -tn

# 查看哪个进程占用了 8080 端口
ss -tlnp | grep 8080

# 查看所有 UDP 监听
ss -ulnp
```

---

### grep

> 在文件或输出中搜索匹配文本

```bash
# 在文件中搜索关键词
grep "error" logfile.txt

# 递归搜索目录下所有文件
grep -rn "TODO" ./src/

# 忽略大小写
grep -i "warning" logfile.txt

# 反向匹配（排除包含 debug 的行）
grep -v "debug" logfile.txt

# 正则表达式搜索
grep -E "error|warning|fatal" logfile.txt

# 只输出匹配的文件名
grep -rl "import torch" ./src/

# 显示匹配行的前后 3 行上下文
grep -C 3 "Exception" logfile.txt

# 统计匹配次数
grep -c "error" logfile.txt

# 配合管道使用
ps aux | grep python
cat /etc/passwd | grep "^root"
```

---

### sed

> 流式编辑器，批量查找替换

```bash
# 替换第一个匹配（每行）
sed 's/old/new/' file.txt

# 全局替换（每行所有匹配）
sed 's/old/new/g' file.txt

# 直接修改文件（原地替换）
sed -i 's/old/new/g' file.txt

# 删除包含特定内容的行
sed '/^#/d' config.txt

# 只打印第 10-20 行
sed -n '10,20p' file.txt

# 在第 3 行后插入内容
sed '3a\new line content' file.txt

# 多条命令
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt
```

---

### awk

> 按列处理文本，轻量数据分析利器

```bash
# 打印第 1 列和第 3 列（默认空格/Tab 分隔）
awk '{print $1, $3}' file.txt

# 指定分隔符（如 CSV 用逗号）
awk -F',' '{print $1, $2}' data.csv

# 条件过滤：打印第 3 列大于 100 的行
awk '$3 > 100' file.txt

# 求和：累加第 2 列
awk '{sum += $2} END {print sum}' data.txt

# 打印行号
awk '{print NR, $0}' file.txt

# 格式化输出
awk '{printf "%-20s %s\n", $1, $2}' file.txt

# 配合管道：统计每个用户的进程数
ps aux | awk '{print $1}' | sort | uniq -c | sort -rn
```

---

### cat

> 查看文件内容、拼接文件

```bash
# 查看文件
cat file.txt

# 显示行号
cat -n file.txt

# 拼接多个文件
cat part1.txt part2.txt > merged.txt

# 追加内容到文件
cat extra.txt >> existing.txt

# 创建文件（Ctrl+D 结束输入）
cat > newfile.txt

# 查看不可见字符（排查编码问题）
cat -A file.txt
```

---

### head / tail

> 查看文件头部/尾部

```bash
# 查看前 20 行
head -n 20 file.txt

# 查看最后 20 行
tail -n 20 file.txt

# 实时追踪日志（非常常用！）
tail -f /var/log/syslog

# 实时追踪并高亮关键词
tail -f logfile.log | grep --color "ERROR"

# 从第 100 行开始显示到末尾
tail -n +100 file.txt
```

---

### sort / uniq

> 排序和去重

```bash
# 排序
sort file.txt

# 数值排序
sort -n numbers.txt

# 逆序
sort -r file.txt

# 按第 2 列数值排序
sort -t',' -k2 -n data.csv

# 去重（必须先排序）
sort file.txt | uniq

# 统计每行出现的次数
sort file.txt | uniq -c | sort -rn

# 只显示重复的行
sort file.txt | uniq -d
```

---

### wc

> 统计行数、单词数、字符数

```bash
# 统计行数
wc -l file.txt

# 统计单词数
wc -w file.txt

# 统计目录下 Python 文件总行数
find . -name "*.py" | xargs wc -l

# 统计管道输出行数
ps aux | grep python | wc -l
```

---

### cut

> 按分隔符或位置截取列

```bash
# 按逗号分隔取第 1 和第 3 列
cut -d',' -f1,3 data.csv

# 按冒号分隔取用户名（/etc/passwd）
cut -d':' -f1 /etc/passwd

# 取每行的第 1-10 个字符
cut -c1-10 file.txt
```

---

### tee

> 同时输出到屏幕和文件

```bash
# 输出同时保存到文件
echo "hello" | tee output.txt

# 追加模式
echo "more" | tee -a output.txt

# 配合管道：运行命令并同时记录日志
python train.py 2>&1 | tee training.log
```

---

### find

> 按条件查找文件

```bash
# 按名称查找
find /path -name "*.py"

# 忽略大小写
find /path -iname "readme*"

# 查找大于 100MB 的文件
find / -size +100M

# 查找最近 7 天修改过的文件
find . -mtime -7

# 查找并删除（慎用！）
find /tmp -name "*.tmp" -delete

# 查找并执行命令
find . -name "*.log" -exec gzip {} \;

# 按类型查找（f=文件 d=目录 l=链接）
find . -type f -name "*.py"

# 排除目录
find . -path ./node_modules -prune -o -name "*.js" -print
```

---

### xargs

> 将标准输入转为命令参数

```bash
# 查找所有 .py 文件并统计行数
find . -name "*.py" | xargs wc -l

# 处理文件名含空格的情况
find . -name "*.txt" -print0 | xargs -0 grep "keyword"

# 并行执行（4 个进程同时运行）
find . -name "*.png" | xargs -P 4 -I {} convert {} -resize 50% {}

# 批量删除 Docker 容器
docker ps -aq | xargs docker rm

# 每次传一个参数
echo "a b c" | xargs -n 1 echo
```

---

### tar

> 打包/解包归档文件

```bash
# 打包并 gzip 压缩
tar -czf archive.tar.gz ./directory/

# 解压 .tar.gz
tar -xzf archive.tar.gz

# 解压到指定目录
tar -xzf archive.tar.gz -C /path/to/dest/

# 查看归档内容（不解压）
tar -tzf archive.tar.gz

# 打包并 xz 压缩（更高压缩率）
tar -cJf archive.tar.xz ./directory/

# 解压 .tar.xz
tar -xJf archive.tar.xz

# 排除特定文件
tar -czf archive.tar.gz --exclude='*.log' ./project/
```

---

### ln

> 创建链接

```bash
# 创建软链接（最常用）
ln -s /path/to/target linkname

# 创建硬链接
ln /path/to/target linkname

# 强制覆盖已存在的链接
ln -sf /new/target linkname

# 查看链接指向
readlink -f linkname
```

---

### chmod / chown

> 修改权限和所有者

```bash
# 给文件添加执行权限
chmod +x script.sh

# 设置权限为 rwxr-xr-x
chmod 755 script.sh

# 递归修改目录权限
chmod -R 644 ./configs/

# 修改文件所有者
chown user:group file.txt

# 递归修改目录所有者
chown -R user:group ./directory/
```

---

### df / du

> 磁盘空间查看

```bash
# 查看磁盘使用情况（人类可读）
df -h

# 查看当前目录大小
du -sh .

# 查看子目录大小并排序
du -sh */ | sort -rh

# 查看前 10 个最大的子目录
du -h --max-depth=1 | sort -rh | head -10

# 查看指定目录
du -sh /var/log
```

---

### ps

> 查看进程快照

```bash
# 查看所有进程（最常用格式）
ps aux

# 查看进程树
ps auxf

# 按内存排序
ps aux --sort=-%mem | head -20

# 按 CPU 排序
ps aux --sort=-%cpu | head -20

# 查找特定进程
ps aux | grep python

# 只看自己的进程
ps ux
```

---

### top / htop

> 实时系统资源监控

```bash
# 启动 top
top

# top 中常用按键：
#   M → 按内存排序
#   P → 按 CPU 排序
#   k → 杀死进程
#   q → 退出

# htop（更好用的替代品）
htop

# 只监控特定用户
top -u username

# 只看指定 PID
top -p 1234,5678

# 非交互式输出一次（脚本中使用）
top -bn1 | head -20
```

---

### kill / killall

> 终止进程

```bash
# 优雅终止（发送 SIGTERM）
kill <PID>

# 强制终止（发送 SIGKILL）
kill -9 <PID>

# 按进程名终止所有匹配进程
killall python

# 强制终止
killall -9 python

# 按名称匹配终止（支持正则）
pkill -f "python train.py"
```

---

### nohup

> 让进程在退出终端后继续运行

```bash
# 后台运行并将输出写入 nohup.out
nohup python train.py &

# 指定输出文件
nohup python train.py > output.log 2>&1 &

# 查看后台进程
jobs -l
```

---

### jobs / bg / fg

> 管理前台/后台任务

```bash
# Ctrl+Z 暂停当前进程
# 查看后台任务
jobs

# 将暂停的任务放到后台继续执行
bg %1

# 将后台任务调回前台
fg %1

# & 直接后台运行
python script.py &
```

---

### systemctl

> 管理 systemd 服务

```bash
# 启动/停止/重启服务
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx

# 查看服务状态
systemctl status nginx

# 开机自启
sudo systemctl enable nginx

# 取消开机自启
sudo systemctl disable nginx

# 查看所有运行中的服务
systemctl list-units --type=service --state=running

# 重新加载配置（不重启）
sudo systemctl reload nginx
```

---

### journalctl

> 查看 systemd 日志

```bash
# 查看某个服务的日志
journalctl -u nginx

# 实时追踪日志
journalctl -u nginx -f

# 查看今天的日志
journalctl --since today

# 查看最近 1 小时的日志
journalctl --since "1 hour ago"

# 查看启动日志
journalctl -b

# 按优先级过滤（err 及以上）
journalctl -p err

# 查看占用磁盘空间
journalctl --disk-usage
```

---

### env / export

> 环境变量管理

```bash
# 查看所有环境变量
env

# 查看特定变量
echo $PATH

# 设置环境变量（当前 shell）
export CUDA_VISIBLE_DEVICES=0,1

# 临时设置环境变量运行命令
CUDA_VISIBLE_DEVICES=0 python train.py

# 持久化：写入 ~/.bashrc 或 ~/.zshrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

### alias

> 给命令起别名

```bash
# 创建别名
alias ll='ls -alh'
alias gs='git status'
alias gp='git push'

# 查看所有别名
alias

# 删除别名
unalias ll

# 持久化：写入 ~/.bashrc 或 ~/.zshrc
echo "alias ll='ls -alh'" >> ~/.bashrc
```

---

### lsblk / mount

> 查看和挂载块设备

```bash
# 查看所有块设备
lsblk

# 显示文件系统信息
lsblk -f

# 挂载设备
sudo mount /dev/sdb1 /mnt/usb

# 卸载
sudo umount /mnt/usb

# 查看当前挂载点
mount | grep "^/dev"
```

---

### tmux

> 终端复用器，断线保持会话

```bash
# 新建会话
tmux new -s mysession

# 断开会话（会话继续运行） → Ctrl+B 然后按 D

# 查看所有会话
tmux ls

# 重新连接会话
tmux attach -t mysession

# 常用快捷键（先按 Ctrl+B）：
#   c → 新建窗口
#   n → 下一个窗口
#   p → 上一个窗口
#   % → 左右分屏
#   " → 上下分屏
#   方向键 → 切换面板

# 杀死会话
tmux kill-session -t mysession
```

---

### watch

> 定时重复执行命令

```bash
# 每 2 秒刷新一次（默认）
watch nvidia-smi

# 每 5 秒刷新
watch -n 5 "df -h"

# 高亮变化部分
watch -d "free -h"

# 监控 GPU 使用情况
watch -n 1 nvidia-smi

# 监控目录文件变化
watch -n 1 "ls -lh /path/to/dir"
```

---

### jq

> 命令行 JSON 解析器

```bash
# 格式化 JSON
echo '{"name":"Tim","age":20}' | jq .

# 提取字段
echo '{"name":"Tim","age":20}' | jq '.name'

# 处理数组
echo '[1,2,3]' | jq '.[]'

# 从 API 响应中提取数据
curl -s https://api.example.com/data | jq '.results[0].name'

# 过滤数组
cat data.json | jq '.[] | select(.score > 90)'

# 构造新的 JSON
cat data.json | jq '{name: .first_name, score: .grade}'
```

---

### strace

> 跟踪进程的系统调用（调试神器）

```bash
# 跟踪命令执行的系统调用
strace ls

# 跟踪正在运行的进程
strace -p <PID>

# 只跟踪文件相关的调用
strace -e trace=file ls

# 只跟踪网络相关的调用
strace -e trace=network curl example.com

# 统计系统调用耗时
strace -c python script.py

# 输出到文件
strace -o trace.log python script.py
```

---

## 实用组合技

```bash
# 查找最大的 10 个文件
find / -type f -exec du -h {} + 2>/dev/null | sort -rh | head -10

# 实时监控日志并过滤关键词
tail -f app.log | grep --line-buffered "ERROR" | tee errors.log

# 批量重命名 .jpeg → .jpg
find . -name "*.jpeg" | while read f; do mv "$f" "${f%.jpeg}.jpg"; done

# 统计代码行数（排除空行和注释）
find . -name "*.py" | xargs grep -v '^\s*$\|^\s*#' | wc -l

# 查看哪个进程占用了某个端口
ss -tlnp | grep :8080

# 远程训练模型不怕断线
tmux new -s train
nohup python train.py 2>&1 | tee train.log
# Ctrl+B, D 断开，之后 tmux attach -t train 重连

# 快速 HTTP 文件服务器
python -m http.server 8080

# 对比两个目录的差异
diff <(ls dir1/) <(ls dir2/)
```