import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { PageList, SortFn, byDateAndAlphabetical } from "../PageList"
import { FullSlug, resolveRelative } from "../../util/path"
import { QuartzPluginData } from "../../plugins/vfile"
import { Date, getDate } from "../Date"
import { i18n } from "../../i18n"
import { concatenateResources } from "../../util/resources"

interface TimelineContentOptions {
  sort?: SortFn
}

interface YearGroup {
  year: number
  months: MonthGroup[]
}

interface MonthGroup {
  month: number
  pages: QuartzPluginData[]
}

export default ((opts?: Partial<TimelineContentOptions>) => {
  const TimelineContent: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData, allFiles, cfg } = props

    // Filter out index pages and pages without dates
    const pagesWithDates = allFiles.filter((file) => {
      const dominated_by_index = file.slug?.endsWith("index")
      const is_tag_page = file.slug?.startsWith("tags/")
      const has_date = file.dates != null
      return !dominated_by_index && !is_tag_page && has_date
    })

    // Sort by date descending
    const sorter = opts?.sort ?? byDateAndAlphabetical(cfg)
    const sortedPages = pagesWithDates.sort(sorter)

    // Group by year and month
    const yearGroups: Map<number, Map<number, QuartzPluginData[]>> = new Map()

    for (const page of sortedPages) {
      const date = getDate(cfg, page)
      if (!date) continue

      const year = date.getFullYear()
      const month = date.getMonth() + 1 // 1-12

      if (!yearGroups.has(year)) {
        yearGroups.set(year, new Map())
      }

      const monthMap = yearGroups.get(year)!
      if (!monthMap.has(month)) {
        monthMap.set(month, [])
      }

      monthMap.get(month)!.push(page)
    }

    // Convert to sorted arrays (years descending, months descending)
    const groupedData: YearGroup[] = Array.from(yearGroups.entries())
      .sort((a, b) => b[0] - a[0]) // years descending
      .map(([year, monthMap]) => ({
        year,
        months: Array.from(monthMap.entries())
          .sort((a, b) => b[0] - a[0]) // months descending
          .map(([month, pages]) => ({ month, pages })),
      }))

    const totalPosts = sortedPages.length

    const monthNames: Record<number, string> = {
      1: "1月",
      2: "2月",
      3: "3月",
      4: "4月",
      5: "5月",
      6: "6月",
      7: "7月",
      8: "8月",
      9: "9月",
      10: "10月",
      11: "11月",
      12: "12月",
    }

    return (
      <div class="timeline-container popover-hint">
        <div class="timeline-header">
          <h1>时间线</h1>
          <p class="timeline-stats">共 {totalPosts} 篇文章</p>
        </div>

        <div class="timeline">
          {groupedData.map(({ year, months }) => (
            <div class="timeline-year" key={year}>
              <h2 class="year-title">{year}</h2>
              <div class="year-content">
                {months.map(({ month, pages }) => (
                  <div class="timeline-month" key={`${year}-${month}`}>
                    <h3 class="month-title">{monthNames[month]}</h3>
                    <ul class="timeline-posts">
                      {pages.map((page) => {
                        const title = page.frontmatter?.title
                        const tags = page.frontmatter?.tags ?? []
                        const date = getDate(cfg, page)

                        return (
                          <li class="timeline-post" key={page.slug}>
                            <div class="post-date">
                              {date && <Date date={date} locale={cfg.locale} />}
                            </div>
                            <div class="post-content">
                              <a
                                href={resolveRelative(fileData.slug!, page.slug!)}
                                class="internal post-title"
                              >
                                {title}
                              </a>
                              {tags.length > 0 && (
                                <ul class="post-tags">
                                  {tags.map((tag) => (
                                    <li key={tag}>
                                      <a
                                        class="internal tag-link"
                                        href={resolveRelative(
                                          fileData.slug!,
                                          `tags/${tag}` as FullSlug,
                                        )}
                                      >
                                        {tag}
                                      </a>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  TimelineContent.css = `
.timeline-container {
  max-width: 800px;
  margin: 0 auto;
}

.timeline-header {
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--lightgray);
}

.timeline-header h1 {
  margin: 0 0 0.5rem 0;
}

.timeline-stats {
  color: var(--gray);
  margin: 0;
}

.timeline {
  position: relative;
}

.timeline::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--lightgray);
}

.timeline-year {
  margin-bottom: 2rem;
  padding-left: 1.5rem;
  position: relative;
}

.year-title {
  font-size: 1.5rem;
  margin: 0 0 1rem 0;
  color: var(--secondary);
  position: relative;
}

.year-title::before {
  content: "";
  position: absolute;
  left: -1.5rem;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 12px;
  height: 12px;
  background: var(--secondary);
  border-radius: 50%;
  border: 2px solid var(--light);
}

.year-content {
  padding-left: 1rem;
}

.timeline-month {
  margin-bottom: 1.5rem;
}

.month-title {
  font-size: 1rem;
  color: var(--gray);
  margin: 0 0 0.75rem 0;
  font-weight: 500;
}

.timeline-posts {
  list-style: none;
  padding: 0;
  margin: 0;
}

.timeline-post {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 0.5rem 0;
  border-bottom: 1px dashed var(--lightgray);
}

.timeline-post:last-child {
  border-bottom: none;
}

.post-date {
  flex-shrink: 0;
  width: 100px;
  font-size: 0.85rem;
  color: var(--gray);
}

.post-content {
  flex: 1;
}

.post-title {
  font-weight: 500;
  display: block;
  margin-bottom: 0.25rem;
}

.post-tags {
  list-style: none;
  padding: 0;
  margin: 0.25rem 0 0 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.post-tags li {
  margin: 0;
}

.post-tags .tag-link {
  font-size: 0.75rem;
  padding: 0.1rem 0.4rem;
  background: var(--highlight);
  border-radius: 4px;
  color: var(--secondary);
}

@media (max-width: 600px) {
  .timeline-post {
    flex-direction: column;
    gap: 0.25rem;
  }

  .post-date {
    width: auto;
  }
}
`

  return TimelineContent
}) satisfies QuartzComponentConstructor
