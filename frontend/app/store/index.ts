import { createStore } from 'vuex';

import detector from './modules/detector';
import type { DetectorState } from './modules/detector';

export interface RootState {
  theme: 'dark' | 'light';
  detector?: DetectorState;
}

export default createStore<RootState>({
  state: () => ({
    theme: 'dark'
  }),

  mutations: {
    SET_THEME(state: RootState, theme: 'light' | 'dark') {
      state.theme = theme;
    }
  },

  actions: {
    initTheme({ commit }) {
      if (import.meta.client) {
        const saved = localStorage.getItem('theme');

        const theme = saved || 'dark';

        commit('SET_THEME', theme);
        document.documentElement.setAttribute('data-theme', theme);
      }
    },
    setTheme({ commit }, theme: 'light' | 'dark') {
      commit('SET_THEME', theme);
      if (import.meta.client) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
      }
    }
  },

  // Подключаем модули
  modules: {
    detector
  }
});
