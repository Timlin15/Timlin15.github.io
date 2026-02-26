驱使我从ToDesk切换到ssh的主要原因是ToDesk不稳定，时不时就会把你踢下线，只有交保护费才能接着用。所有就转到了ssh和远程桌面。

简单来说，ssh可以用本地的VSCode访问远程服务器的文件和命令行，只要你的需求是编写代码和运行命令行工具，你可以在有人使用ToDesk的时候利用SSH同步修改代码，不需要因为有人使用ToDesk就完全没办法使用服务器。
而有了SSH，因为amax安装的是Ubuntu24.04，自带远程桌面的支持，我已经设置好了，理论上你只要
- 按下 `Win + R`，输入 `mstsc` 回车。
- **Computer** 处填写 `amax` 的 Tailscale IP（即 `100.91.146.72`）。
- 点击 **Connect**，在弹出的窗口中输入amax的用户名和密码，即amax和Amax1979！即可连上
在由于我用的不是Windows，我没有验证这个步骤，我在Linux环境下是能正常连接的。
远程桌面和ToDesk是大致相同的，如果你是ToDesk的会员可以不用设置。对于普通版，远程桌面不需要担心被踢下线且有更清晰的画质。
## 安装步骤
首先，在你的电脑上[安装Tailscale](https://tailscale.com/download)，注册账号，然后用amax的访问链接。
设置好tailscale就可以通过ssh连接上amax了。见 https://tailscale.com/kb/1022/install-windows ，应该不是很复杂，注意连接的时候VPN最好不要用tun模式。
我用的基本上很稳定，偶尔会掉线，建议搭配使用**tmux**使用，防止工作丢失。

然后依照
- 按下 `Win + R`，输入 `mstsc` 回车。
- **Computer** 处填写 `amax` 的 Tailscale IP（即 `100.91.146.72`）。
- 点击 **Connect**，在弹出的窗口中输入amax的用户名和密码即可连上
![image.png](https://typora-1344509263.cos.ap-guangzhou.myqcloud.com/markdown/20260128153331637.png)



