import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import readingTime from "reading-time"

export default (() => {
  const TotalWordCount: QuartzComponent = ({
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    // Calculate total word count from all files
    let totalWords = 0
    let totalArticles = 0

    for (const file of allFiles) {
      if (file.text) {
        const { words } = readingTime(file.text)
        totalWords += words
        totalArticles++
      }
    }

    // Format number with commas for readability
    const formattedWords = totalWords.toLocaleString("zh-CN")

    return (
      <div class={classNames(displayClass, "total-word-count")}>
        <span class="word-count-badge">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
          </svg>
          {formattedWords} 字
        </span>
        <span class="article-count-badge">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          {totalArticles} 篇文章
        </span>
      </div>
    )
  }

  TotalWordCount.css = `
.total-word-count {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin: 0.3rem 0 0.8rem 0;
}

.word-count-badge,
.article-count-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  font-size: 0.8rem;
  color: var(--darkgray);
  background: transparent;
  border: 1px solid var(--lightgray);
  border-radius: 0.3rem;
  transition: all 0.2s ease;
}

.word-count-badge:hover,
.article-count-badge:hover {
  border-color: var(--gray);
}

.word-count-badge svg,
.article-count-badge svg {
  flex-shrink: 0;
  opacity: 0.7;
}
`

  return TotalWordCount
}) satisfies QuartzComponentConstructor
