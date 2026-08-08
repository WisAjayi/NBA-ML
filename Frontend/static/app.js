document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.getElementById('nav-toggle');
    const siteNav = document.getElementById('site-nav');

    if (navToggle && siteNav) {
        navToggle.addEventListener('click', () => {
            const isOpen = siteNav.classList.toggle('open');
            navToggle.setAttribute('aria-expanded', String(isOpen));
        });

        siteNav.querySelectorAll('a').forEach((link) => {
            link.addEventListener('click', () => {
                siteNav.classList.remove('open');
                navToggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    const allPlayers = window.ALL_PLAYERS || [];

    function playerLabel(p) {
        return `${p.first} ${p.last.replace(/_/g, ' ')} (${p.team})`;
    }

    document.querySelectorAll('[data-players-datalist]').forEach((datalist) => {
        allPlayers.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = playerLabel(p);
            datalist.appendChild(opt);
        });
    });

    document.querySelectorAll('[data-roster-picker]').forEach((picker) => {
        const teamSelect = picker.querySelector('[data-team-select]');
        const playerSelect = picker.querySelector('[data-player-select]');
        const firstInput = picker.querySelector('[data-first-input]');
        const lastInput = picker.querySelector('[data-last-input]');
        const quickSearch = picker.querySelector('[data-quick-search]');

        if (!teamSelect || !playerSelect) return;

        function populatePlayers(team, preselect) {
            const players = (window.ROSTERS && window.ROSTERS[team]) || [];
            playerSelect.innerHTML = '';

            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = players.length ? 'Select a player' : 'No players found for this team';
            playerSelect.appendChild(placeholder);

            players.forEach((p) => {
                const opt = document.createElement('option');
                opt.value = `${p.first}||${p.last}`;
                opt.textContent = `${p.first} ${p.last.replace(/_/g, ' ')}`;
                playerSelect.appendChild(opt);
            });

            playerSelect.disabled = players.length === 0;

            if (preselect) {
                playerSelect.value = preselect;
                const [first, last] = preselect.split('||');
                if (firstInput) firstInput.value = first || '';
                if (lastInput) lastInput.value = last || '';
            } else {
                if (firstInput) firstInput.value = '';
                if (lastInput) lastInput.value = '';
            }
        }

        teamSelect.addEventListener('change', () => populatePlayers(teamSelect.value));

        playerSelect.addEventListener('change', () => {
            const [first, last] = playerSelect.value.split('||');
            if (firstInput) firstInput.value = first || '';
            if (lastInput) lastInput.value = last || '';
        });

        if (quickSearch) {
            quickSearch.addEventListener('input', () => {
                const match = allPlayers.find((p) => playerLabel(p) === quickSearch.value);
                if (!match) return;
                teamSelect.value = match.team;
                populatePlayers(match.team, `${match.first}||${match.last}`);
            });
        }

        if (teamSelect.value) {
            populatePlayers(teamSelect.value);
        }
    });
});
