export * from './stores';

export type BtnUi = 'primary' | 'secondary' | 'link' | 'dark' | 'ghost';

export type BtnSize = 's' | 'm' | 'l';

export type IconUi = 'primary' | 'secondary' | 'dark';

export type IconSize = 's' | 'm' | 'l';

export interface BaseError {
  description?: string;
}

export interface UploadedFile {
  id: number;
  file: File;
  name: string;
  size: number;
  progress: number;
  status: 'loading' | 'ready' | 'error';
}

export interface SpanResult {
  startChar: number;
  endChar: number;
  text: string;
  avgConfidence: number;
}

export interface DetectionResult {
  verdict: string;
  confidence: number;
  aiPercentage: number;
  riskLevel: string;
  spans: SpanResult[];
  explanation: string;
  technicalConsensus: string;
  judgeAgreement: number;
  needsHumanReview: boolean;
  reviewReason: string;
}

export interface GatewayHealth {
  status: string;
  service: string;
  calibrator: boolean;
  calibratorUrl: string;
  detail: string | null;
}

// Сырые данные от Бэкенда (snake_case)
export interface RawSpanResult {
  start_char: number;
  end_char: number;
  text: string;
  avg_confidence: number;
}

export interface RawDetectionResult {
  verdict: 'AI' | 'HUMAN' | 'MIXED';
  confidence: number;
  ai_percentage: number;
  risk_level: 'low' | 'medium' | 'high';
  spans: RawSpanResult[];
  explanation: string;
  technical_consensus: string;
  judge_agreement: number;
  needs_human_review: boolean;
  review_reason: string;
}

export interface RawGatewayHealth {
  status: string;
  service: string;
  calibrator: boolean;
  calibrator_url: string;
  detail: string | null;
}
