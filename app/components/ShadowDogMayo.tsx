"use client";

import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const CAT_SRC = "/cat/mayo.png";

/** 옥견 그림자 결인이 잡힌 것으로 볼 최소 연속 프레임 */
const GYOKKEN_STREAK_ON = 14;
const GYOKKEN_STREAK_OFF = 10;

type Lm = { x: number; y: number; z?: number };

type HandLandmarkerHandle = {
  detectForVideo: (
    video: HTMLVideoElement,
    timestamp: number,
  ) => { landmarks: Lm[][] };
  close: () => void;
};

function dist2(a: Lm, b: Lm) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** 검지~새끼: 펼침 */
function isFourFingerExtended(lm: Lm[], tip: number, pip: number) {
  const w = lm[0];
  return dist2(lm[tip], w) > dist2(lm[pip], w) * 1.06;
}

function isThumbExtendedOut(lm: Lm[]) {
  const w = lm[0];
  return dist2(lm[4], w) > dist2(lm[3], w) * 1.04;
}

/** 엄지는 접힌 편(옆 손: 검·중만 길게) */
function isThumbFoldedIn(lm: Lm[]) {
  const w = lm[0];
  return dist2(lm[4], w) <= dist2(lm[3], w) * 1.12;
}

/**
 * 아래 손(그림자의 주둥이): 검지·중지만 펼치고, 약지·새끼·엄지는 접음.
 * 두 손가락 방향이 비슷해야 함.
 */
function isSnoutHand(lm: Lm[]): boolean {
  const idx = isFourFingerExtended(lm, 8, 6);
  const mid = isFourFingerExtended(lm, 12, 10);
  const ring = isFourFingerExtended(lm, 16, 14);
  const pinky = isFourFingerExtended(lm, 20, 18);
  if (!idx || !mid || ring || pinky) return false;
  if (!isThumbFoldedIn(lm)) return false;

  const w = lm[0];
  const v8 = { x: lm[8].x - w.x, y: lm[8].y - w.y };
  const v12 = { x: lm[12].x - w.x, y: lm[12].y - w.y };
  const m =
    Math.hypot(v8.x, v8.y) * Math.hypot(v12.x, v12.y);
  if (m < 1e-5) return false;
  const cos = (v8.x * v12.x + v8.y * v12.y) / m;
  return cos > 0.82;
}

/**
 * 위 손(그림자의 귀): 엄지만 세우고 나머지 네 손가락은 굽힘.
 * 엄지 끝이 손목보다 위이거나, 다른 손끝들 중 가장 위쪽에 가깝게.
 */
function isEarHandThumbUp(lm: Lm[]): boolean {
  if (!isThumbExtendedOut(lm)) return false;

  const w = lm[0];
  const thumbTip = lm[4];
  const minOtherY = Math.min(lm[8].y, lm[12].y, lm[16].y, lm[20].y);
  const thumbMostUpright =
    thumbTip.y < w.y - 0.012 || thumbTip.y <= minOtherY + 0.035;

  if (!thumbMostUpright) return false;

  return (
    !isFourFingerExtended(lm, 8, 6) &&
    !isFourFingerExtended(lm, 12, 10) &&
    !isFourFingerExtended(lm, 16, 14) &&
    !isFourFingerExtended(lm, 20, 18)
  );
}

function wristsClose(a: Lm[], b: Lm[], max: number) {
  return dist2(a[0], b[0]) <= max;
}

/**
 * 만화 속 옥견 그림자: 양손 겹침 + 한 손은 엄지만 위, 다른 손은 검·중만 길게.
 * 감지 순서가 바뀔 수 있어 양쪽 배치를 모두 검사.
 */
function isGyokkenTwinShadowPose(pair: Lm[][]): boolean {
  if (pair.length < 2) return false;
  const A = pair[0];
  const B = pair[1];
  if (!wristsClose(A, B, 0.48)) return false;

  const match1 = isEarHandThumbUp(A) && isSnoutHand(B);
  const match2 = isEarHandThumbUp(B) && isSnoutHand(A);
  return match1 || match2;
}

export default function ShadowDogMayo() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const rafRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const gyokkenOnStreak = useRef(0);
  const gyokkenOffStreak = useRef(0);
  const gyokkenActiveRef = useRef(false);

  const [step, setStep] = useState<"welcome" | "live">("welcome");
  const [camError, setCamError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<
    "off" | "loading" | "ready" | "error"
  >("off");
  const [handCount, setHandCount] = useState(0);
  const [gyokkenActive, setGyokkenActive] = useState(false);

  const [floatPos, setFloatPos] = useState({ x: 50, y: 50 });

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const v = videoRef.current;
    if (v) v.srcObject = null;
  }, []);

  const startCamera = async () => {
    setCamError(null);
    const video = videoRef.current;
    if (!video) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      setStep("live");
      setModelState("loading");
    } catch (e) {
      console.error(e);
      setCamError(
        "카메라를 켤 수 없습니다. 브라우저에서 카메라 권한을 허용했는지 확인해 주세요.",
      );
    }
  };

  useEffect(() => {
    return () => stopStream();
  }, [stopStream]);

  /** 옥견 결인이 유지되는 동안만 무작위 이동 */
  useEffect(() => {
    if (step !== "live" || !gyokkenActive) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const jump = () => {
      setFloatPos({
        x: 10 + Math.random() * 80,
        y: 10 + Math.random() * 80,
      });
    };

    const schedule = () => {
      const delay = 1400 + Math.random() * 2200;
      timer = setTimeout(() => {
        if (cancelled) return;
        jump();
        schedule();
      }, delay);
    };

    jump();
    schedule();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [step, gyokkenActive]);

  useEffect(() => {
    if (step !== "live") return;

    let cancelled = false;
    let landmarker: HandLandmarkerHandle | null = null;

    const stopLoop = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };

    const setGyokkenUi = (active: boolean) => {
      if (gyokkenActiveRef.current === active) return;
      gyokkenActiveRef.current = active;
      setGyokkenActive(active);
    };

    let prevHandCount = -1;

    const loop = () => {
      const video = videoRef.current;
      if (!video || !landmarker) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      if (video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const result = landmarker.detectForVideo(video, performance.now());
      const lms = result.landmarks;
      const n = lms.length;

      if (n !== prevHandCount) {
        prevHandCount = n;
        setHandCount(n);
      }

      const gyokken = n >= 2 && isGyokkenTwinShadowPose(lms);

      if (gyokken) {
        gyokkenOffStreak.current = 0;
        gyokkenOnStreak.current += 1;
        if (gyokkenOnStreak.current >= GYOKKEN_STREAK_ON) {
          setGyokkenUi(true);
        }
      } else {
        gyokkenOnStreak.current = 0;
        gyokkenOffStreak.current += 1;
        if (gyokkenOffStreak.current >= GYOKKEN_STREAK_OFF) {
          setGyokkenUi(false);
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    (async () => {
      try {
        const { FilesetResolver, HandLandmarker } = await import(
          "@mediapipe/tasks-vision"
        );
        if (cancelled) return;
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        landmarker = (await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        })) as HandLandmarkerHandle;
        if (cancelled) {
          landmarker.close();
          return;
        }
        setModelState("ready");
        rafRef.current = requestAnimationFrame(loop);
      } catch (e) {
        console.error(e);
        if (!cancelled) setModelState("error");
      }
    })();

    return () => {
      cancelled = true;
      stopLoop();
      landmarker?.close();
      gyokkenOnStreak.current = 0;
      gyokkenOffStreak.current = 0;
      gyokkenActiveRef.current = false;
      setGyokkenActive(false);
      setHandCount(0);
    };
  }, [step]);

  const statusLine = (() => {
    if (gyokkenActive) return "玉犬 · 옥견 그림자 결인 포착";
    if (handCount === 0) return "양손으로 옥견 그림자를 만들어 주세요";
    if (handCount === 1)
      return "한 손 더 — 참고처럼 위아래로 겹쳐 주세요";
    return "엄지만 위로 세운 손 + 검지·중지만 펼친 손 — 맞춰 주세요";
  })();

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-8">
      <header className="text-center">
        <p className="text-xs font-medium tracking-wide text-violet-600 dark:text-violet-400">
          十種影法術 · 십종영법술 모티브 (팬 메이드)
        </p>
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
          작중 옥견(玉犬)을 부릴 때 쓰는{" "}
          <strong>양손 그림자 결인</strong>을 웹캠으로 맞추면 고양이(
          <code className="text-xs">public/cat/mayo.png</code>)가 뜹니다.
          한 손만 보이거나 다른 모양이면 나오지 않습니다.
        </p>
      </header>

      {step === "welcome" && (
        <div className="flex flex-col items-center gap-4 rounded-2xl border border-zinc-200 bg-white p-10 dark:border-zinc-800 dark:bg-zinc-950">
          <p className="max-w-md text-center text-sm text-zinc-600 dark:text-zinc-400">
            그림자를 비추려면 먼저 &quot;영(影)&quot;이 들어올 통로를 열어야
            합니다. 양손이 동시에 잡히도록 카메라 거리를 두는 것이 좋습니다.
          </p>
          <button
            type="button"
            onClick={startCamera}
            className="rounded-full bg-zinc-900 px-6 py-3 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
          >
            그림자 매개 열기
          </button>
          {camError && (
            <p className="max-w-md text-center text-sm text-red-600 dark:text-red-400">
              {camError}
            </p>
          )}
        </div>
      )}

      <div className={step === "live" ? "block" : "contents"}>
        <div
          className={
            step === "live"
              ? "relative aspect-video w-full overflow-hidden rounded-2xl border border-zinc-200 bg-black shadow-lg dark:border-zinc-800"
              : "sr-only"
          }
        >
          <video
            ref={videoRef}
            className={
              step === "live"
                ? "relative z-0 h-full w-full scale-x-[-1] object-cover"
                : "h-1 w-1 opacity-0"
            }
            playsInline
            muted
          />

          {step === "live" && gyokkenActive && modelState === "ready" && (
            <div
              className="pointer-events-none absolute z-10 w-[min(36vw,200px)] max-w-[42%] select-none transition-[left,top] duration-[950ms] ease-in-out"
              style={{
                left: `${floatPos.x}%`,
                top: `${floatPos.y}%`,
                transform: "translate(-50%, -50%)",
                aspectRatio: "1",
              }}
            >
              <Image
                src={CAT_SRC}
                alt="메이"
                fill
                className="object-contain drop-shadow-[0_6px_20px_rgba(0,0,0,0.45)]"
                sizes="(max-width: 768px) 36vw, 200px"
                priority
              />
            </div>
          )}

          {step === "live" && modelState === "loading" && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-black/50 text-sm text-white">
              그림자 감지 술식 불러오는 중…
            </div>
          )}
          {step === "live" && modelState === "error" && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-red-950/90 p-4 text-center text-sm text-red-100">
              그림자 감지 술식을 불러오지 못했습니다.
            </div>
          )}
          {step === "live" && modelState === "ready" && (
            <div className="pointer-events-none absolute left-3 top-3 z-20 max-w-[min(92%,280px)]">
              <span
                className={`inline-block rounded-full px-3 py-1 text-xs backdrop-blur-sm ${gyokkenActive
                  ? "bg-amber-500/95 font-medium text-amber-950"
                  : "bg-black/55 text-white"
                  }`}
              >
                {statusLine}
              </span>
            </div>
          )}
        </div>
      </div>

      {step === "live" && (
        <div className="rounded-xl border border-zinc-200 bg-white p-4 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300">
          <p className="font-medium text-zinc-900 dark:text-zinc-100">
            옥견 그림자 결인 (참고)
          </p>
          <ul className="mt-2 list-inside list-disc space-y-1">
            <li>
              <strong>한 손</strong>: 엄지만 위로 세우고, 검지·중지·약지·새끼는
              굽혀 위쪽 손등을 덮듯이.
            </li>
            <li>
              <strong>다른 손</strong>: 검지와 중지만 곧게 펴서 &quot;주둥이&quot;
              느낌으로, 엄지·약지·새끼는 접기.
            </li>
            <li>
              두 손을 만화처럼 겹치고, 손목끼리 가깝게 맞추면 인식이 잘 붙습니다.
            </li>
          </ul>
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-500">
            『주술회전』 비공식 2차 창작 UI입니다. 실제 작중 결인·효과와 다를 수
            있습니다.
          </p>
        </div>
      )}
    </div>
  );
}
