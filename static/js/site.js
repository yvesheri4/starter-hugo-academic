const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('#main-nav');
if (menu && nav) {
  document.documentElement.classList.add('has-menu');
  menu.hidden = false;
  const close = () => { menu.setAttribute('aria-expanded', 'false'); nav.classList.remove('is-open'); };
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    nav.classList.toggle('is-open', open);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') { close(); menu.focus(); } });
  nav.addEventListener('click', e => { if (e.target.closest('a')) close(); });
}
document.querySelectorAll('[data-animation]').forEach(button => {
  const img = document.getElementById(button.getAttribute('aria-controls'));
  button.hidden = false;
  button.addEventListener('click', () => {
    const playing = button.getAttribute('aria-pressed') !== 'true';
    img.src = playing ? button.dataset.animation : button.dataset.poster;
    img.alt = playing ? 'Animated two-stream number distribution from normalized time 0 to 40' : 'Two-stream number distribution at normalized time 40';
    button.setAttribute('aria-pressed', String(playing));
    button.textContent = playing ? 'Stop animation' : 'Play animation';
  });
});
const search = document.querySelector('#publication-search');
if (search) {
  search.closest('.publication-search').hidden = false;
  search.addEventListener('input', () => {
    const query = search.value.trim().toLocaleLowerCase();
    let count = 0;
    document.querySelectorAll('.publication-group').forEach(group => {
      let visible = 0;
      group.querySelectorAll('.publication-row').forEach(row => {
        row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
        if (!row.hidden) visible++;
      });
      group.hidden = visible === 0;
      count += visible;
    });
    document.querySelector('#publication-count').textContent = `${count} record${count === 1 ? '' : 's'} found`;
  });
}
