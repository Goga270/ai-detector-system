import type { Module } from 'vuex';

import type { RootState } from '../index';

export interface DetectionResult {
  success: boolean;
  analysis: {
    ai_probability: number;
    human_probability: number;
    prediction: 'AI Generated' | 'Human';
    confidence: 'Low' | 'Medium' | 'High';
    text_length: { characters: number; words: number };
    features: {
      punctuation_ratio: number;
      stopword_ratio: number;
      unique_word_ratio: number;
    };
  };
  interpretation: {
    is_likely_ai: boolean;
    confidence_score: number;
  };
}

export interface DetectorState {
  currentText: string;
  result: DetectionResult | null;
  loading: boolean;
  error: string | null;
}

const detector: Module<DetectorState, RootState> = {
  namespaced: true,
  state: () => ({
    currentText: '',
    result: null,
    loading: false,
    error: null
  }),

  mutations: {
    SET_TEXT(state, text: string) {
      state.currentText = text;
    },
    SET_RESULT(state, result: DetectionResult) {
      state.result = result;
      state.error = null;
    },
    SET_LOADING(state, status: boolean) {
      state.loading = status;
    },
    SET_ERROR(state, message: string) {
      state.error = message;
      state.result = null;
    }
  },

  actions: {
    async analyzeText({ commit, state }) {
      if (!state.currentText.trim()) return;

      commit('SET_LOADING', true);
      try {
        const data = await $fetch<DetectionResult>('/api/analyze', {
          method: 'POST',
          body: { text: state.currentText }
        });

        if (data.success) {
          commit('SET_RESULT', data);
        } else {
          commit('SET_ERROR', 'Ошибка анализа на стороне модели');
        }
      } catch (err) {
        commit('SET_ERROR', err.statusMessage || 'Ошибка сервера');
      } finally {
        commit('SET_LOADING', false);
      }
    }
  }
};

export default detector;
