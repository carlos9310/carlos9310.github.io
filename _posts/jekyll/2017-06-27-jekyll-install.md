---
layout: post
title: Jekyll Ubuntu 环境初始化
category: jekyll
description: 
---

在发布到github pages 时，需要本地预览效果，这时需要安装本的jekyll环境，这里说明安装过程中的主要步骤以及相应的解决办法。

## Install CURL

```
apt-get install curl
```

## Install RVM 

```
gpg2 --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
curl -sSL https://get.rvm.io | bash -s stable
```

## source rvm environment

```
source /home/user/.rvm/scripts/rvm
```

## Install Ruby Dependencies 

```
rvm requirements
```

## Install ruby

```
rvm isntall 2.4.0
```

## set Default ruby version

```
rvm use 2.4.0 --default
```

## install jekyll

```
gem install jekyll kramdown
```
gem 包含在rubygem中，安装完ruby后，自然会安装


## 依赖安装问题：
```
/home/user/.rvm/rubies/ruby-2.4.0/lib/ruby/2.4.0/rubygems/core_ext/kernel_require.rb:55:in `require': cannot load such file -- bundler (LoadError)
	from /home/user/.rvm/rubies/ruby-2.4.0/lib/ruby/2.4.0/rubygems/core_ext/kernel_require.rb:55:in `require'
	from /home/user/.rvm/gems/ruby-2.4.0/gems/jekyll-3.5.0/lib/jekyll/plugin_manager.rb:46:in `require_from_bundler'
	from /home/user/.rvm/gems/ruby-2.4.0/gems/jekyll-3.5.0/exe/jekyll:9:in `<top (required)>'
	from /home/user/.rvm/gems/ruby-2.4.0/bin/jekyll:22:in `load'
	from /home/user/.rvm/gems/ruby-2.4.0/bin/jekyll:22:in `<main>'
	from /home/user/.rvm/gems/ruby-2.4.0/bin/ruby_executable_hooks:15:in `eval'
	from /home/user/.rvm/gems/ruby-2.4.0/bin/ruby_executable_hooks:15:in `<main>'

```
执行以下命令

```
gem install bundler
gem install github-pages
```

## JavaScript 运行环境问题 

```
Jekyll 3.4.3 | Error:  Could not find a JavaScript runtime. See https://github.com/rails/execjs for a list of available runtimes.
```

安装nodejs

```
sudo apt-get install nodejs
```

### 问题描述

```
  Liquid Exception: undefined method `map' for false:FalseClass Did you mean? tap in /_layouts/page.html

```

* [问题参考](https://github.com/github/pages-gem/issues/351) 使用jekyll新建一个主页，解决此问题

```
 Deprecation: The 'gems' configuration option has been renamed to 'plugins'. Please update your config file accordingly.
```
配置文件_config.yml中，使用了 plugins 的配置项，应该是用plugins替换掉gems。


## 调用github 无法获取token的问题

* 问题描述
```
GitHub Metadata: No GitHub API authentication could be found. Some fields may be missing or have incorrect data
```

* 解决方法
```
JEKYLL_GITHUB_TOKEN=<your token> jekyll serve -H 172.16.42.71
```

* 参考资料
[No GitHub API authentication" error](https://github.com/github/pages-gem/issues/399#issuecomment-361091215)

## 评论插件

* [为博客添加 Gitalk 评论插件](https://www.jianshu.com/p/78c64d07124d)

## 参考资料

* [How To Install Ruby on Rails on Ubuntu](https://tecadmin.net/install-ruby-on-rails-on-ubuntu/)
* [Jekyll中国](http://jekyllcn.com/)