#!/bin/bash

echo "🔄 正在同步到 GitHub..."

msg="auto sync at $(date +'%Y-%m-%d %H:%M:%S')"

git add .
git commit -m "$msg"
git push origin main

echo "✅ 同步完成！"
