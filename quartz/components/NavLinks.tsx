import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { resolveRelative } from "../util/path"

interface NavLink {
  title: string
  slug: string
}

interface NavLinksOptions {
  links: NavLink[]
}

const defaultOptions: NavLinksOptions = {
  links: [],
}

export default ((userOpts?: Partial<NavLinksOptions>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const NavLinks: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    return (
      <nav class="nav-links">
        {opts.links.map((link) => (
          <a
            href={resolveRelative(fileData.slug!, link.slug)}
            class="internal"
            data-no-popover="true"
          >
            <h2>{link.title}</h2>
          </a>
        ))}
      </nav>
    )
  }

  NavLinks.css = `
.nav-links {
  display: flex;
  flex-direction: column;
}

.nav-links a.internal {
  background-color: transparent !important;
  background: none !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  text-align: left;
  cursor: pointer;
  color: var(--dark);
  display: flex;
  align-items: center;
}

.nav-links a.internal h2 {
  font-size: 1rem;
  display: inline-block;
  margin: 0;
}

.nav-links a.internal:hover h2 {
  color: var(--tertiary);
}
`

  return NavLinks
}) satisfies QuartzComponentConstructor
