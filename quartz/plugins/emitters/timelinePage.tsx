import { QuartzEmitterPlugin } from "../types"
import { QuartzComponentProps } from "../../components/types"
import HeaderConstructor from "../../components/Header"
import BodyConstructor from "../../components/Body"
import { pageResources, renderPage } from "../../components/renderPage"
import { defaultProcessedContent } from "../vfile"
import { FullPageLayout } from "../../cfg"
import { FullSlug, pathToRoot } from "../../util/path"
import { defaultListPageLayout, sharedPageComponents } from "../../../quartz.layout"
import { TimelineContent } from "../../components"
import { write } from "./helpers"
import { QuartzPluginData } from "../vfile"

interface TimelinePageOptions extends FullPageLayout {
  sort?: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

export const TimelinePage: QuartzEmitterPlugin<Partial<TimelinePageOptions>> = (userOpts) => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultListPageLayout,
    pageBody: TimelineContent({ sort: userOpts?.sort }),
    ...userOpts,
  }

  const { head: Head, header, beforeBody, pageBody, afterBody, left, right, footer: Footer } = opts
  const Header = HeaderConstructor()
  const Body = BodyConstructor()

  return {
    name: "TimelinePage",
    getQuartzComponents() {
      return [
        Head,
        Header,
        Body,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async *emit(ctx, content, resources) {
      const allFiles = content.map((c) => c[1].data)
      const cfg = ctx.cfg.configuration

      const slug = "timeline" as FullSlug
      const externalResources = pageResources(pathToRoot(slug), resources)

      // Create a virtual page for the timeline
      const [tree, file] = defaultProcessedContent({
        slug,
        frontmatter: { title: "时间线", tags: [] },
      })

      const componentData: QuartzComponentProps = {
        ctx,
        fileData: file.data,
        externalResources,
        cfg,
        children: [],
        tree,
        allFiles,
      }

      const renderedContent = renderPage(cfg, slug, componentData, opts, externalResources)

      yield write({
        ctx,
        content: renderedContent,
        slug,
        ext: ".html",
      })
    },
    async *partialEmit(ctx, content, resources, _changeEvents) {
      // Always rebuild the timeline page when any content changes
      // since the timeline shows all posts
      const allFiles = content.map((c) => c[1].data)
      const cfg = ctx.cfg.configuration

      const slug = "timeline" as FullSlug
      const externalResources = pageResources(pathToRoot(slug), resources)

      const [tree, file] = defaultProcessedContent({
        slug,
        frontmatter: { title: "时间线", tags: [] },
      })

      const componentData: QuartzComponentProps = {
        ctx,
        fileData: file.data,
        externalResources,
        cfg,
        children: [],
        tree,
        allFiles,
      }

      const renderedContent = renderPage(cfg, slug, componentData, opts, externalResources)

      yield write({
        ctx,
        content: renderedContent,
        slug,
        ext: ".html",
      })
    },
  }
}
