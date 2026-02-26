import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

interface SocialLink {
  icon: string // SVG path
  href: string
  title: string
  viewBox?: string // optional custom viewBox, default "0 0 24 24"
}

interface Options {
  links: SocialLink[]
}

const defaultOptions: Options = {
  links: [],
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const SocialLinks: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "social-links")}>
        {opts.links.map((link) => (
          <a href={link.href} title={link.title} target="_blank" rel="noopener noreferrer">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox={link.viewBox ?? "0 0 24 24"}
              fill="currentColor"
            >
              <path d={link.icon} />
            </svg>
          </a>
        ))}
      </div>
    )
  }

  SocialLinks.css = `
.social-links {
  display: flex;
  gap: 0.8rem;
  margin: 0.5rem 0;
}

.social-links a {
  color: var(--gray);
  transition: color 0.2s ease;
}

.social-links a:hover {
  color: var(--secondary);
}

.social-links svg {
  display: block;
}
`

  return SocialLinks
}) satisfies QuartzComponentConstructor
