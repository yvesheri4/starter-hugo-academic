# Yves Heri — Research profile

A responsive, multi-page Hugo website for research, projects, talks, publications, personal photographs, and a blog. No JavaScript framework or Hugo theme modules are required.

## Preview

Use Hugo 0.147.9 (the version pinned in `netlify.toml`):

```sh
hugo server
```

Open http://localhost:1313. Build for production with `hugo --gc --minify`.

## Updating the profile

- `data/profile.json`: research areas, selected publication, presentations, education, and awards.
- `layouts/index.html`: homepage highlights.
- `layouts/partials/`: shared header, footer, metadata, cards, and contact details.
- `profile/post/`: blog posts in Markdown; `profile/project/` and `profile/talk/`: project and presentation pages.
- `layouts/about/single.html`: biography and academic background.
- `data/gallery.json` and `static/gallery/`: the original personal gallery, optimized for the web.
- `config/_default/menus.yaml`: site navigation.
- `static/css/profile.css`: responsive styles.
- `static/uploads/resume.pdf`: downloadable CV. Update its visible date in the footer when replacing it.
- `static/portrait.jpeg`: existing portrait from the original website.
- `config/_default/config.yaml`: site title, description, domain, and content directory.

Content is based on the supplied October 6, 2026 CV, the existing biography, and all 17 Google Scholar records inspected on October 6, 2026. Journal articles, preprints, conference papers/abstracts, and earlier work are distinguished. PASCHEN-1D is listed by Scholar in Computer Physics Communications 329, 110404 (December 2026 issue), superseding the CV's under-review label; the issue date is explicit and its open preprint is linked. Earlier records without a verified venue link to their Scholar entries. The original May 2025 CV is superseded by the latest supplied PDF; no PDF content has been edited.

Sources: https://scholar.google.com/citations?user=sYU3KYIAAAAJ&hl=en ; https://arxiv.org/abs/2608.07721 ; https://arxiv.org/abs/2607.24590 ; https://www.openscience.fr/Modelisation-d-un-reseau-de-distribution-BT-avec-cables-preassembles .

## Original content and hosting

The original Wowchemy content remains in `content/` for reference, but Hugo publishes from `profile/`. Demo publications, projects, posts, and talks are therefore excluded from the built website. The original Go module file is retained for reference; the site configuration no longer imports theme modules.

Netlify continues to build into `public/`. Deploy previews use their own base URL; production uses the configured www.yves-heri.com domain. `static/_redirects` maps old demo URLs to the corresponding archives while preserving working posts, projects, and talks. The custom 404 page handles other missing pages.

Fonts load from Google Fonts, with local system fallbacks. Navigation, citations, and contact links work without JavaScript. The beam drawing is a conceptual SVG illustration, explicitly labeled as such; it is not research data.

## Review checks

Before deployment, build with Hugo, inspect the desktop and mobile pages, confirm all navigation anchors and the PDF download, and review the scientific wording against the latest CV. The redesign has not been merged or published automatically.


## Blog workflow

Create a draft with `hugo new post/my-post/index.md`. Edit the Markdown file under `profile/post/`; set its title, description, date, and category. Preview drafts with `hugo server -D`. Set `draft: false` when ready to include the post in a production build. The posts archive and RSS feed at `/post/index.xml` update automatically. The introductory post is newly drafted for local review; it is not imported historical writing.

Project and talk pages use Markdown with JSON front matter (also supported by Hugo). Talk dates with only a known month use the first day internally for sorting; the page displays only the known month and year. Do not infer presentation roles from coauthorship of conference records.

The gallery retains the original 15 photos, with descriptive captions and optimized WebP copies. Original assets and legacy Wowchemy content are preserved in the repository. All text files and JSON imports must be read and written explicitly as UTF-8.
