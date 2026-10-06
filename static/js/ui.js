(function () {
    'use strict';

    document.querySelectorAll('[data-tabs]').forEach(function (tabGroup) {
        var tabs = Array.prototype.slice.call(tabGroup.querySelectorAll('[role="tab"]'));
        var panels = Array.prototype.slice.call(tabGroup.querySelectorAll('[role="tabpanel"]'));

        function activateTab(tab, moveFocus) {
            var panelId = tab.getAttribute('aria-controls');
            var selectedPanel = panels.find(function (panel) {
                return panel.id === panelId;
            });

            if (!selectedPanel) return;

            tabs.forEach(function (groupTab) {
                var selected = groupTab === tab;
                groupTab.setAttribute('aria-selected', String(selected));
                groupTab.tabIndex = selected ? 0 : -1;
            });

            panels.forEach(function (panel) {
                panel.hidden = panel !== selectedPanel;
            });

            if (moveFocus) tab.focus();
        }

        tabs.forEach(function (tab, index) {
            tab.addEventListener('click', function () {
                activateTab(tab, false);
            });

            tab.addEventListener('keydown', function (event) {
                var nextIndex;
                if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                    nextIndex = (index + 1) % tabs.length;
                } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                    nextIndex = (index + tabs.length - 1) % tabs.length;
                } else if (event.key === 'Home') {
                    nextIndex = 0;
                } else if (event.key === 'End') {
                    nextIndex = tabs.length - 1;
                } else {
                    return;
                }

                event.preventDefault();
                activateTab(tabs[nextIndex], true);
            });
        });

        var initiallySelected = tabs.find(function (tab) {
            return tab.getAttribute('aria-selected') === 'true';
        });
        if (initiallySelected) activateTab(initiallySelected, false);
    });

    document.addEventListener('click', function (event) {
        var dismissButton = event.target.closest('[data-dismiss]');
        if (!dismissButton) return;

        var alert = dismissButton.closest('.alert');
        if (alert) alert.remove();
    });
}());
