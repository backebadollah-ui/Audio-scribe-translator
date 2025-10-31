export interface Language {
  code: string;
  name: string;
}

export enum AppStatus {
  IDLE,
  RECORDING,
  PROCESSING,
  TRANSLATING,
  FINISHED,
  ERROR,
}
