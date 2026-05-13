import type { Module } from 'vuex';

import type { RootState } from '~/store';

export interface User {
  id: string;
  name: string;
  email: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

const auth: Module<AuthState, RootState> = {
  namespaced: true,
  state: () => ({
    user: null,
    token: null,
    loading: false,
    error: null
  }),

  mutations: {
    SET_USER(state, payload: { user: User; token: string }) {
      state.user = payload.user;
      state.token = payload.token;
      state.error = null;
    },
    SET_LOADING(state, status: boolean) {
      state.loading = status;
    },
    SET_ERROR(state, error: string | null) {
      state.error = error;
    },
    LOGOUT(state) {
      state.user = null;
      state.token = null;
    }
  },

  actions: {
    async login({ commit }, payload) {
      commit('SET_LOADING', true);
      commit('SET_ERROR', null);

      try {
        // Имитация задержки сети (1.5 секунды)
        await new Promise((resolve) => setTimeout(resolve, 1500));

        // Простейшая мок-проверка
        if (payload.email === 'test@test.com' && payload.password === 'password') {
          const mockData = {
            user: { id: '1', name: 'Иван Иванов', email: payload.email },
            token: 'mock-jwt-token-12345'
          };

          commit('SET_USER', mockData);

          if (import.meta.client) {
            localStorage.setItem('token', mockData.token);
          }

          return true; // Успех
        } else {
          throw new Error('Неверный email или пароль');
        }
      } catch (e) {
        commit('SET_ERROR', e.message);

        return false;
      } finally {
        commit('SET_LOADING', false);
      }
    }
  }
};

export default auth;
