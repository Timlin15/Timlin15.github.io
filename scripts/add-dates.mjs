#!/usr/bin/env node

/**
 * 自动为 Markdown 文件添加 date 字段到 frontmatter
 * 从 Obsidian 库读取文件的真实创建时间
 *
 * 用法: node scripts/add-dates.mjs [--dry-run]
 *   --dry-run: 只显示会修改的文件，不实际修改
 */

import fs from 'fs'
import path from 'path'
import { execSync } from 'child_process'

const CONTENT_DIR = './content'
const OBSIDIAN_VAULT = '/home/matrix/Documents/Obsidian Vault'
const DRY_RUN = process.argv.includes('--dry-run')

// 使用 stat 命令获取文件创建时间（Birth time）
function getFileBirthTime(filePath) {
  try {
    // Linux: stat --format="%W" 返回 birth time 的 Unix 时间戳
    const birthTime = execSync(`stat --format="%W" "${filePath}"`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim()

    if (birthTime && birthTime !== '0' && birthTime !== '-') {
      const date = new Date(parseInt(birthTime) * 1000)
      return formatDate(date)
    }
  } catch {
    // 忽略错误
  }
  return null
}

// 使用 stat 命令获取文件修改时间
function getFileModifyTime(filePath) {
  try {
    const mtime = execSync(`stat --format="%Y" "${filePath}"`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim()

    if (mtime) {
      const date = new Date(parseInt(mtime) * 1000)
      return formatDate(date)
    }
  } catch {
    // 忽略错误
  }
  return null
}

function formatDate(date) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function getFileDate(contentFilePath) {
  // 计算对应的 Obsidian 库文件路径
  const relativePath = path.relative(CONTENT_DIR, contentFilePath)
  const obsidianFilePath = path.join(OBSIDIAN_VAULT, relativePath)

  // 优先从 Obsidian 库读取创建时间
  if (fs.existsSync(obsidianFilePath)) {
    const birthTime = getFileBirthTime(obsidianFilePath)
    if (birthTime) {
      return { date: birthTime, source: 'Obsidian 创建时间' }
    }

    const mtime = getFileModifyTime(obsidianFilePath)
    if (mtime) {
      return { date: mtime, source: 'Obsidian 修改时间' }
    }
  }

  // 回退到 content 目录的时间
  const birthTime = getFileBirthTime(contentFilePath)
  if (birthTime) {
    return { date: birthTime, source: 'content 创建时间' }
  }

  const mtime = getFileModifyTime(contentFilePath)
  if (mtime) {
    return { date: mtime, source: 'content 修改时间' }
  }

  // 最后回退到 Node.js 的 fs.statSync
  const stats = fs.statSync(contentFilePath)
  return { date: formatDate(stats.mtime), source: 'Node.js mtime' }
}

function hasDateField(content) {
  // 检查 frontmatter 中是否已有 date 相关字段
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/)
  if (!frontmatterMatch) return false

  const frontmatter = frontmatterMatch[1]
  // 检查是否有 date, created, published 等字段
  return /^(date|created|published):/m.test(frontmatter)
}

function addDateToFrontmatter(content, date) {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/)

  if (frontmatterMatch) {
    // 已有 frontmatter，在其中添加 date
    const frontmatter = frontmatterMatch[1]
    const newFrontmatter = `date: ${date}\n${frontmatter}`
    return content.replace(/^---\n[\s\S]*?\n---/, `---\n${newFrontmatter}\n---`)
  } else {
    // 没有 frontmatter，创建新的
    return `---\ndate: ${date}\n---\n\n${content}`
  }
}

function processFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8')

  if (hasDateField(content)) {
    return { skipped: true, reason: '已有 date 字段' }
  }

  const { date, source } = getFileDate(filePath)
  const newContent = addDateToFrontmatter(content, date)

  if (!DRY_RUN) {
    fs.writeFileSync(filePath, newContent, 'utf-8')
  }

  return { skipped: false, date, source }
}

function walkDir(dir) {
  const files = []
  const entries = fs.readdirSync(dir, { withFileTypes: true })

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      // 跳过隐藏目录和特殊目录
      if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
        files.push(...walkDir(fullPath))
      }
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      files.push(fullPath)
    }
  }

  return files
}

// 主程序
console.log(DRY_RUN ? '=== 预览模式 (不会修改文件) ===' : '=== 开始添加日期 ===')
console.log(`Obsidian 库路径: ${OBSIDIAN_VAULT}`)
console.log()

const mdFiles = walkDir(CONTENT_DIR)
let modified = 0
let skipped = 0

for (const file of mdFiles) {
  const relativePath = path.relative(CONTENT_DIR, file)
  const result = processFile(file)

  if (result.skipped) {
    skipped++
  } else {
    modified++
    console.log(`${DRY_RUN ? '[预览]' : '[已添加]'} ${relativePath}`)
    console.log(`         -> date: ${result.date} (${result.source})`)
  }
}

console.log()
console.log(`完成! 修改: ${modified} 个文件, 跳过: ${skipped} 个文件`)

if (DRY_RUN && modified > 0) {
  console.log()
  console.log('提示: 移除 --dry-run 参数以实际修改文件')
}
