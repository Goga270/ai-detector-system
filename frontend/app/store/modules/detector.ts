import type { Module } from 'vuex';

import type {
  BaseError,
  DetectionResult,
  GatewayHealth,
  RawDetectionResult,
  RawGatewayHealth,
  RawSpanResult,
  SpanResult
} from '~/types';

import type { RootState } from '../index';

export interface DetectorState {
  currentText: string;
  result: DetectionResult | null;
  loading: boolean;
  error: string | null;
  health: GatewayHealth | null;
}

// Функция маппинга (Anti-Corruption Layer)
function mapDetectionResult(data: RawDetectionResult): DetectionResult {
  return {
    verdict: data.verdict || 'UNKNOWN',
    confidence: data.confidence || 0,
    aiPercentage: data.ai_percentage || 0,
    riskLevel: data.risk_level || 'low',
    explanation: data.explanation || '',
    technicalConsensus: data.technical_consensus || '',
    judgeAgreement: data.judge_agreement || 0,
    needsHumanReview: data.needs_human_review || false,
    reviewReason: data.review_reason || '',
    spans: (data.spans || []).map(
      (span: RawSpanResult): SpanResult => ({
        startChar: span.start_char,
        endChar: span.end_char,
        text: span.text,
        avgConfidence: span.avg_confidence
      })
    )
  };
}

function mapGatewayHealth(data: RawGatewayHealth): GatewayHealth {
  return {
    status: data.status || 'error',
    service: data.service || 'unknown',
    calibrator: data.calibrator || false,
    calibratorUrl: data.calibrator_url || '',
    detail: data.detail || null
  };
}

function getDynamicMock(inputText: string): RawDetectionResult {
  const textLength = inputText.length;

  const start = Math.floor(textLength * 0.3);

  const end = Math.floor(textLength * 0.7);

  return {
    verdict: 'MIXED',
    confidence: 0.88,
    ai_percentage: 45.5,
    risk_level: 'medium',
    explanation: 'Наши детекторы обнаружили существенные стилистические аномалии в центральной части текста.',
    technical_consensus: 'BERT указывает на генерацию, DetectGPT сомневается.',
    judge_agreement: 0.75,
    needs_human_review: true,
    review_reason: 'Смешанный стиль письма',
    spans:
      textLength > 20
        ? [
            {
              start_char: start,
              end_char: end,
              text: inputText.substring(start, end),
              avg_confidence: 0.92
            }
          ]
        : []
  };
}

const detector: Module<DetectorState, RootState> = {
  namespaced: true,
  state: () => ({
    currentText: '',
    result: null,
    loading: false,
    error: null,
    health: null
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
    },
    SET_HEALTH: (state, val) => (state.health = val),
    CLEAR_ALL: (state) => {
      state.result = null;
      state.currentText = '';
      state.error = null;
    }
  },

  actions: {
    async checkSystemHealth({ commit }) {
      const config = useRuntimeConfig();

      try {
        const data = await $fetch<RawGatewayHealth>(`${config.public.apiBaseUrl}/gateway/health`);

        commit('SET_HEALTH', mapGatewayHealth(data));
      } catch (err: unknown) {
        const e = err as BaseError;

        console.error('Система недоступна', e.description);
      }
    },
    async analyzeText({ commit, state }) {
      if (!state.currentText.trim()) return;

      const config = useRuntimeConfig();

      commit('SET_LOADING', true);
      commit('SET_RESULT', null);
      commit('SET_ERROR', null);
      try {
        if (config.public.useMocks) {
          await new Promise((resolve) => setTimeout(resolve, 1500));
          const rawMock = getDynamicMock(state.currentText);

          commit('SET_RESULT', mapDetectionResult(rawMock));

          return;
        }

        const data = await $fetch<RawDetectionResult>(`${config.public.apiBaseUrl}/gateway/detect/text`, {
          method: 'POST',
          body: { text: state.currentText }
        });

        commit('SET_RESULT', mapDetectionResult(data));
      } catch (err: unknown) {
        const e = err as BaseError;

        commit('SET_ERROR', e.description || 'Ошибка сервера');
      } finally {
        commit('SET_LOADING', false);
      }
    },
    async analyzeFile({ commit }, file: File) {
      const config = useRuntimeConfig();

      commit('SET_LOADING', true);
      commit('SET_ERROR', null);
      commit('SET_RESULT', null);

      try {
        if (config.public.useMocks) {
          await new Promise((resolve) => setTimeout(resolve, 1500));
          const mock = getDynamicMock('Текст из PDF...');

          commit('SET_RESULT', mapDetectionResult(mock));

          return;
        }
        const formData = new FormData();

        formData.append('file', file);

        const data = await $fetch<RawDetectionResult>(`${config.public.apiBaseUrl}/gateway/detect/pdf`, {
          method: 'POST',
          body: formData
        });

        commit('SET_RESULT', mapDetectionResult(data));
      } catch (err: unknown) {
        const e = err as BaseError;

        commit('SET_ERROR', e.description || 'Ошибка при обработке PDF');
      } finally {
        commit('SET_LOADING', false);
      }
    }
  }
};

export default detector;
