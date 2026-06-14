import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

#建立與初始化 RAG 知識庫
def setup_vector_db():
    print("初始化 RAG 向量資料庫")
    
    #根據你的 JSON 指標設計的衛教知識庫
    knowledge_base_data = [
        {
            "content": "當ROM狀態為'insufficient'(活動度不足)時，代表患者未達到目標角度。回饋原則：1. 鼓勵患者在無痛範圍內盡量增加動作幅度。2. 給予具體方向（例如：『您的手可以試著再抬高一點』）。",
            "metadata": {"category": "rom", "trigger": "insufficient"}
        },
        {
            "content": "當偵測到局部關節偏差 (per_joint_deviation) 時，代表特定關節偏離標準軌跡。回饋原則：必須明確指出是哪一個關節（如：左手肘），並提醒患者注意該部位的穩定度。",
            "metadata": {"category": "per_joint_deviation", "trigger": "deviation"}
        },
        {
            "content": "當 peak (頂峰期) 狀態為 'angle_drop_detected' 時，代表患者在動作最高點無法維持穩定。回饋原則：提醒患者在動作最高點稍微停頓 1 到 2 秒，感受肌肉收縮，不要急著放下來。",
            "metadata": {"category": "phase", "trigger": "angle_drop"}
        },
        {
            "content": "當 descent (下降期) 狀態為 'abnormal_speed' 時，代表缺乏離心收縮控制力。回饋原則：提醒患者放下肢體時要放慢速度，在心裡默數 3 秒慢慢放下。",
            "metadata": {"category": "phase", "trigger": "abnormal_speed"}
        },
        {
            "content": "當速度與平順度狀態為 'tremor_detected' (異常顫抖) 時，代表肌肉疲勞或控制力不足。回饋原則：語氣溫和，提醒患者若覺得痠痛無力可先休息，安全第一。",
            "metadata": {"category": "velocity", "trigger": "tremor"}
        },
        {
            "content": "當發生軀幹偏移 (trunk_lean) 時，代表患者利用身體歪斜代償。回饋原則：溫和點出錯誤，提醒患者『保持核心穩定，背部挺直』，避免腰背受傷。",
            "metadata": {"category": "compensation", "trigger": "trunk_lean"}
        },
        {
            "content": "當 'shrugging_detected' 為 true (聳肩) 時，代表使用了脖子力量代償。回饋原則：提醒患者將肩膀放鬆下沉，遠離耳朵，避免肩頸痠痛。",
            "metadata": {"category": "compensation", "trigger": "shrugging"}
        },
        {
            "content": "當 'veto_triggered' 為 true 時，軌跡極度異常，觸發 Veto。回饋原則：語氣中立，『系統發現動作軌跡差異較大，這次先不計分。請放輕鬆，確認動作要領再試一次』。",
            "metadata": {"category": "veto", "trigger": "anomaly"}
        }
    ]

    documents = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in knowledge_base_data
    ]

    # 使用 HuggingFace 模型將文字轉換為向量
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 建立 Chroma 資料庫
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory="./rehab_chroma_db" 
    )

    # 回傳檢索器 (k=3 代表如果病患同時犯多個錯，我們最多抓 3 條相關的衛教規則給 LLM 參考)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

#解析 JSON 數據，提取錯誤項目
def parse_system_errors(data):
    
    #從系統回傳的複雜 JSON 中，自動抓取被扣分的項目
    errors = []
    
    #優先檢查防線指標 (Veto)
    if data.get("metadata", {}).get("veto_triggered", False):
        return ["動作軌跡極端變形 (觸發 Veto 拒絕計分)"]
        
    primary = data.get("primary_metrics", {})
    
    #檢查 ROM
    if primary.get("rom", {}).get("status") == "insufficient":
        errors.append("關節活動度(ROM)不足")
        
    #檢查關節偏差
    for joint_data in primary.get("per_joint_deviation", []):
        if joint_data.get("max_deviation_degree", 0) > 15.0: #假設大於15度視為明顯偏差
            errors.append(f"{joint_data.get('joint')} 嚴重偏差 {joint_data.get('max_deviation_degree')} 度")
            
    #檢查階段性分析
    phase = primary.get("phase_specific_analysis", {})
    if phase.get("peak") == "angle_drop_detected":
        errors.append("頂峰期角度掉落 (未維持住)")
    if phase.get("descent") == "abnormal_speed":
        errors.append("下降期速度異常 (過快)")
        
    #檢查速度與顫抖
    if primary.get("velocity_and_smoothness", {}).get("status") == "tremor_detected":
        errors.append("動作出現異常顫抖")
        
    #檢查對稱性與代償
    sym = primary.get("symmetry_and_compensation", {})
    if sym.get("trunk_lean_degree", 0) > 10.0: #假設大於10度為軀幹偏移
        errors.append(f"軀幹向旁偏移 {sym.get('trunk_lean_degree')} 度")
    if sym.get("shrugging_detected"):
        errors.append("出現聳肩代償行為")
        
    return errors

#結合 RAG 與 Ollama 生成最終回饋
def generate_rehab_feedback(system_json, retriever):
    print("\n--- 生成 AI 治療師回饋 ---")
    
    action_name = system_json.get("metadata", {}).get("action_target", "未知動作")
    score = system_json.get("metadata", {}).get("overall_score", 0)
    
    #解析錯誤清單
    error_list = parse_system_errors(system_json)
    
    if not error_list:
        return "您的動作非常標準，請繼續保持這個完美的節奏！"
        
    error_str = "、".join(error_list)
    print(f"系統偵測到的問題: {error_str}")
    
    #用錯誤字串去知識庫搜尋相關的衛教原則
    search_query = f"{action_name} 發生以下問題：{error_str}"
    retrieved_docs = retriever.invoke(search_query)
    
    #將撈出來的多條規則合併成一段大字串
    rag_context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
    print(f"成功撈取對應知識庫規則，準備生成建議...")

    #定義給 LLM 的 Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["action", "score", "errors", "rag_context"],
        template="""
        你是一位專業、有耐心且充滿同理心的物理治療師。
        
        【病患目前的動作數據】
        - 動作名稱：{action}
        - 系統綜合評分：{score} 分
        - 偵測到的具體問題：{errors}
        
        【臨床衛教指引 (請根據以下指引給予建議)】
        {rag_context}
        
        【你的任務】
        請綜合上述「病患的問題」與「臨床衛教指引」，寫一段給病患的口語化反饋。
        
        要求條件：
        1. 字數控制在 80 到 100 字以內。
        2. 語氣溫暖、鼓勵，像真人對話。
        3. 必須明確指出病患的問題（如：聳肩、角度不夠、顫抖等），並告訴他具體該怎麼調整。
        4. 全文請使用繁體中文。
        """
    )

    #初始化本地端的 Ollama 模型
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.6)
    
    #呼叫 LLM
    print("Ollama 模型思考中...")
    chain = prompt_template | llm
    
    response = chain.invoke({
        "action": action_name,
        "score": score,
        "errors": error_str,
        "rag_context": rag_context
    })
    
    return response.content

if __name__ == "__main__":
    #啟動時先載入資料庫
    my_retriever = setup_vector_db()
    
    #測試資料(暫時)
    json_input = """
    {
      "metadata": {
        "action_target": "shoulder_abduction",
        "repetition_count": 1,
        "started_at": 1610000.0,
        "ended_at": 1610005.5,
        "duration_s": 5.5,
        "overall_score": 78.5,
        "pass_status": "Fail",
        "veto_triggered": False
      },
      "primary_metrics": {
        "rom": {
          "target_angle": 150.0,
          "achieved_max_angle": 120.5,
          "status": "insufficient"
        },
        "per_joint_deviation": [
          {
            "joint": "left_elbow",
            "max_deviation_degree": 18.2,
            "average_deviation_degree": 12.0
          }
        ],
        "phase_specific_analysis": {
          "ascent": "normal",
          "peak": "angle_drop_detected",
          "descent": "abnormal_speed"
        },
        "velocity_and_smoothness": {
          "jerk_value": 25.4,
          "status": "tremor_detected"
        },
        "symmetry_and_compensation": {
          "trunk_lean_degree": 15.0,
          "shrugging_detected": True,
          "symmetry_offset": 8.5
        }
      },
      "auxiliary_metrics": {
        "dtw_global_similarity": 0.82,
        "anomaly_aware_flag": False,
        "tracking_quality": {
          "occlusion_detected": False,
          "low_confidence_joints": []
        }
      }
    }
    """

    #將字串轉換為 Python 字典
    system_data = json.loads(json_input)
    
    #產生回饋
    final_feedback = generate_rehab_feedback(system_data, my_retriever)
    
    print("【AI 治療師給病患的建議】")
    print(final_feedback)