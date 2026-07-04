(ns kami.gen.hybrid-test
  (:require [clojure.test :refer [deftest testing is]]
            [comfyui.node :as node]
            [comfyui.workflow :as wf]
            [kami.gen.hybrid :as hybrid]
            [kami.gen.hybrid.workflow :as hwf]
            [kami.gen.hybrid.nodes :as nodes]
            [kami.gen.hybrid.texture :as texture]
            [vrm.glb :as glb])
  (:import [javax.imageio ImageIO]
           [java.io File]))

(defn- tmp-dir []
  (let [f (File. (str (System/getProperty "java.io.tmpdir") "/kami-gen-hybrid-test-" (System/nanoTime)))]
    (.mkdirs f)
    (.getAbsolutePath f)))

;; ── costume-prompt->workflow is well-formed per comfyui-clj's own schema ──

(deftest workflow-is-well-formed
  (let [{:keys [workflow targets]}
        (hwf/costume-prompt->workflow "penguin kigurumi, chibi, gray/black body, yellow beak"
                                       {:width 64 :height 64 :seed 42})
        reg (node/registry (nodes/pack texture/mock-execute-texture (tmp-dir)))
        result (wf/validate reg workflow)]
    (testing "structurally a comfyui-clj API-format workflow"
      (is (map? workflow))
      (is (every? string? (keys workflow)))
      (is (every? #(contains? % :class_type) (vals workflow))))
    (testing "targets are the 3 SaveTextureImage nodes"
      (is (= #{"save_albedo" "save_normal" "save_roughness"} (set targets))))
    (testing "passes comfyui-clj's own validate against the real node registry"
      (is (:valid? result) (pr-str (:errors result))))
    (testing "checkpoint -> CLIP encode -> sampler -> VAE decode chain shape, per channel"
      (doseq [c ["albedo" "normal" "roughness"]]
        (is (= "KSampler" (get-in workflow [(str "sampler_" c) :class_type])))
        (is (= "VAEDecode" (get-in workflow [(str "decode_" c) :class_type])))
        (is (= "SaveTextureImage" (get-in workflow [(str "save_" c) :class_type])))
        (is (= [(str "sampler_" c) 0] (get-in workflow [(str "decode_" c) :inputs :samples])))
        (is (= [(str "decode_" c) 0] (get-in workflow [(str "save_" c) :inputs :images])))))
    (testing "rejects a malformed workflow (dangling link) — validate isn't a no-op"
      (let [bad (assoc-in workflow ["decode_albedo" :inputs :samples] ["no-such-node" 0])]
        (is (false? (:valid? (wf/validate reg bad))))))))

;; ── full round-trip against the mock texture executor ────────────────

(deftest generate-round-trip-with-mock-executor
  (let [out-dir (tmp-dir)
        result (hybrid/generate
                {:base :race/human
                 :costume-prompt "penguin kigurumi, chibi, gray/black body, yellow beak"
                 :seed 42}
                {:execute-texture texture/mock-execute-texture
                 :output-dir out-dir
                 :base-resolution 64})
        {:keys [body workflow textures vrm]} result]

    (testing ":body is a bare humanoid — no clothing/costume geometry"
      (is (seq (:parts body)))
      (is (not-any? #(= "clothing" (:name %)) (:parts body)))
      (is (some #(= "body" (:name %)) (:parts body)))
      (is (some #(= "head" (:name %)) (:parts body)))
      (is (seq (:blendshape-targets body)))
      (is (seq (get-in body [:skeleton :bones]))))

    (testing ":workflow is well-formed per comfyui-clj's own schema"
      (let [reg (node/registry (nodes/pack texture/mock-execute-texture out-dir))]
        (is (:valid? (wf/validate reg workflow)))))

    (testing ":textures are real PNG files sized to the body's UV texture slot"
      (doseq [[k path] textures]
        (is (.exists (File. ^String path)) (str k " " path))
        (let [img (ImageIO/read (File. ^String path))]
          (is (= 64 (.getWidth img)) k)
          (is (= 64 (.getHeight img)) k)))
      (testing "albedo isn't a degenerate single-color image (real per-pixel checkerboard)"
        (let [img (ImageIO/read (File. ^String (:albedo textures)))
              colors (set (for [x (range 0 64 4) y (range 0 64 4)] (.getRGB img x y)))]
          (is (> (count colors) 1) (pr-str colors)))))

    (testing "same seed+prompt -> same textures (deterministic mock, not random)"
      (let [out-dir2 (tmp-dir)
            result2 (hybrid/generate
                     {:base :race/human
                      :costume-prompt "penguin kigurumi, chibi, gray/black body, yellow beak"
                      :seed 42}
                     {:execute-texture texture/mock-execute-texture
                      :output-dir out-dir2
                      :base-resolution 64})
            img1 (ImageIO/read (File. ^String (get-in result [:textures :albedo])))
            img2 (ImageIO/read (File. ^String (get-in result2 [:textures :albedo])))]
        (is (= (.getRGB img1 5 5) (.getRGB img2 5 5)))
        (is (= (.getRGB img1 40 12) (.getRGB img2 40 12)))))

    (testing ":vrm is a real GLB/VRM file with the textures actually attached"
      (is (some? vrm))
      (is (.exists (File. ^String (:path vrm))))
      (is (pos? (count (:glb-bytes vrm))))
      (let [parsed (glb/parse-glb (:glb-bytes vrm))]
        (is (some? (:json parsed)))
        (is (some? (:bin parsed))))
      (let [gltf (get-in vrm [:doc :gltf])
            materials (:materials gltf)
            skin-mat (some #(when (= "skin" (:name %)) %) materials)]
        (is (some? skin-mat) "a :skin material exists")
        (is (some? (get-in skin-mat [:pbrMetallicRoughness :baseColorTexture :index]))
            "albedo attached via baseColorTexture")
        (is (some? (:normalTexture skin-mat)) "normal map attached via normalTexture")
        (is (some? (get-in skin-mat [:pbrMetallicRoughness :metallicRoughnessTexture :index]))
            "roughness attached via metallicRoughnessTexture")
        ;; textures actually resolve to embedded images with real byte data,
        ;; not just dangling indices
        (let [tex-idx (get-in skin-mat [:pbrMetallicRoughness :baseColorTexture :index])
              img-idx (get-in gltf [:textures tex-idx :source])
              image (get-in gltf [:images img-idx])
              bv (get-in gltf [:bufferViews (:bufferView image)])]
          (is (= "image/png" (:mimeType image)))
          (is (pos? (:byteLength bv)))))
      (testing "humanoid bone mapping present (hips at minimum)"
        (is (some #(= :hips (:bone %)) (get-in vrm [:doc :humanoid :human-bones])))))))

;; ── real-execute-texture fails loudly without a live endpoint ─────────

(deftest real-execute-texture-without-endpoint-throws
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no live ComfyUI/diffusion endpoint"
                         (texture/real-execute-texture {} {:prompt "x" :width 8 :height 8
                                                            :seed 0 :channel :albedo}))))

;; ── unsupported base fails loudly rather than silently guessing ──────

(deftest unsupported-base-throws
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported :base"
                         (hybrid/bare-body :race/elf))))
