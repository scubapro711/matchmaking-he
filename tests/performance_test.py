#!/usr/bin/env python3
"""
בדיקות ביצועים למערכת השידוכים
"""
import time
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import logging

# הגדרת לוגינג
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000", token: str = "demo_token_123"):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.results = []
        
    def test_single_match_request(self, candidate_id: str, max_results: int = 10) -> Dict:
        """בדיקת בקשת התאמה יחידה"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/match/search",
                headers=self.headers,
                json={
                    "candidate_id": candidate_id,
                    "max_results": max_results,
                    "min_score": 0.3
                },
                timeout=300  # 5 דקות timeout
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # במילישניות
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "candidate_id": candidate_id,
                    "success": True,
                    "response_time_ms": response_time,
                    "server_time_ms": data.get("search_time_ms", 0),
                    "matches_found": data.get("total_found", 0),
                    "avg_score": np.mean([m["total_score"] for m in data.get("matches", [])]) if data.get("matches") else 0,
                    "max_score": max([m["total_score"] for m in data.get("matches", [])]) if data.get("matches") else 0,
                    "error": None
                }
            else:
                return {
                    "candidate_id": candidate_id,
                    "success": False,
                    "response_time_ms": response_time,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            return {
                "candidate_id": candidate_id,
                "success": False,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    def test_concurrent_requests(self, candidate_ids: List[str], max_workers: int = 5) -> List[Dict]:
        """בדיקת בקשות מקבילות"""
        logger.info(f"מתחיל בדיקת {len(candidate_ids)} בקשות מקבילות עם {max_workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # שליחת כל הבקשות
            future_to_candidate = {
                executor.submit(self.test_single_match_request, candidate_id): candidate_id 
                for candidate_id in candidate_ids
            }
            
            # איסוף התוצאות
            for future in as_completed(future_to_candidate):
                candidate_id = future_to_candidate[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"הושלם: {candidate_id} - {result['success']}")
                except Exception as e:
                    logger.error(f"שגיאה עבור {candidate_id}: {e}")
                    results.append({
                        "candidate_id": candidate_id,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def test_load_performance(self, num_requests: int = 50) -> Dict:
        """בדיקת עומס על המערכת"""
        logger.info(f"מתחיל בדיקת עומס עם {num_requests} בקשות")
        
        # קבלת רשימת מועמדים
        try:
            df = pd.read_csv("data/unified_candidates.csv")
            candidate_ids = df["מזהה"].sample(min(num_requests, len(df))).tolist()
        except Exception as e:
            logger.error(f"שגיאה בטעינת מועמדים: {e}")
            return {"error": str(e)}
        
        # בדיקות עם רמות concurrency שונות
        concurrency_levels = [1, 3, 5, 10]
        results = {}
        
        for concurrency in concurrency_levels:
            if concurrency > len(candidate_ids):
                continue
                
            logger.info(f"בדיקה עם concurrency {concurrency}")
            test_candidates = candidate_ids[:min(20, len(candidate_ids))]  # מגביל ל-20 לכל בדיקה
            
            start_time = time.time()
            concurrent_results = self.test_concurrent_requests(test_candidates, concurrency)
            total_time = time.time() - start_time
            
            successful_results = [r for r in concurrent_results if r["success"]]
            
            results[f"concurrency_{concurrency}"] = {
                "total_requests": len(concurrent_results),
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / len(concurrent_results) * 100,
                "total_time_seconds": total_time,
                "avg_response_time_ms": np.mean([r["response_time_ms"] for r in successful_results]) if successful_results else 0,
                "median_response_time_ms": np.median([r["response_time_ms"] for r in successful_results]) if successful_results else 0,
                "p95_response_time_ms": np.percentile([r["response_time_ms"] for r in successful_results], 95) if successful_results else 0,
                "avg_matches_found": np.mean([r.get("matches_found", 0) for r in successful_results]) if successful_results else 0,
                "avg_score": np.mean([r.get("avg_score", 0) for r in successful_results]) if successful_results else 0,
                "throughput_requests_per_second": len(successful_results) / total_time if total_time > 0 else 0
            }
        
        return results
    
    def test_algorithm_quality(self, sample_size: int = 100) -> Dict:
        """בדיקת איכות האלגוריתם"""
        logger.info(f"מתחיל בדיקת איכות אלגוריתם עם {sample_size} דוגמאות")
        
        try:
            df = pd.read_csv("data/unified_candidates.csv")
            candidate_ids = df["מזהה"].sample(min(sample_size, len(df))).tolist()
        except Exception as e:
            logger.error(f"שגיאה בטעינת מועמדים: {e}")
            return {"error": str(e)}
        
        quality_results = []
        
        for candidate_id in candidate_ids[:20]:  # מגביל ל-20 לבדיקת איכות
            result = self.test_single_match_request(candidate_id, max_results=10)
            if result["success"]:
                quality_results.append(result)
        
        if not quality_results:
            return {"error": "לא נמצאו תוצאות מוצלחות"}
        
        # ניתוח איכות
        scores = [r["avg_score"] for r in quality_results if r.get("avg_score", 0) > 0]
        max_scores = [r["max_score"] for r in quality_results if r.get("max_score", 0) > 0]
        matches_counts = [r["matches_found"] for r in quality_results]
        
        return {
            "total_tested": len(quality_results),
            "avg_score_mean": np.mean(scores) if scores else 0,
            "avg_score_std": np.std(scores) if scores else 0,
            "max_score_mean": np.mean(max_scores) if max_scores else 0,
            "max_score_std": np.std(max_scores) if max_scores else 0,
            "avg_matches_found": np.mean(matches_counts),
            "matches_found_std": np.std(matches_counts),
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 0.8]) / len(scores) * 100 if scores else 0,
                "good": len([s for s in scores if 0.6 <= s < 0.8]) / len(scores) * 100 if scores else 0,
                "fair": len([s for s in scores if 0.4 <= s < 0.6]) / len(scores) * 100 if scores else 0,
                "poor": len([s for s in scores if s < 0.4]) / len(scores) * 100 if scores else 0
            }
        }
    
    def generate_performance_report(self, output_file: str = "performance_report.json"):
        """יצירת דוח ביצועים מקיף"""
        logger.info("מתחיל יצירת דוח ביצועים מקיף")
        
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "load_performance": self.test_load_performance(50),
            "algorithm_quality": self.test_algorithm_quality(100)
        }
        
        # שמירת הדוח
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"דוח ביצועים נשמר ב: {output_file}")
        return report
    
    def create_performance_visualizations(self, report: Dict, output_dir: str = "performance_charts"):
        """יצירת גרפים של ביצועי המערכת"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # הגדרת פונט עברי
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
        
        # גרף 1: זמני תגובה לפי רמת concurrency
        if "load_performance" in report and isinstance(report["load_performance"], dict):
            concurrency_data = []
            for key, value in report["load_performance"].items():
                if key.startswith("concurrency_") and isinstance(value, dict):
                    concurrency_level = int(key.split("_")[1])
                    concurrency_data.append({
                        "concurrency": concurrency_level,
                        "avg_response_time": value.get("avg_response_time_ms", 0),
                        "p95_response_time": value.get("p95_response_time_ms", 0),
                        "success_rate": value.get("success_rate", 0),
                        "throughput": value.get("throughput_requests_per_second", 0)
                    })
            
            if concurrency_data:
                df_perf = pd.DataFrame(concurrency_data)
                
                # גרף זמני תגובה
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.plot(df_perf["concurrency"], df_perf["avg_response_time"], 'o-', label="Average Response Time")
                ax1.plot(df_perf["concurrency"], df_perf["p95_response_time"], 's-', label="95th Percentile")
                ax1.set_xlabel("Concurrency Level")
                ax1.set_ylabel("Response Time (ms)")
                ax1.set_title("Response Time vs Concurrency")
                ax1.legend()
                ax1.grid(True)
                
                # גרף throughput
                ax2.plot(df_perf["concurrency"], df_perf["throughput"], 'o-', color='green')
                ax2.set_xlabel("Concurrency Level")
                ax2.set_ylabel("Requests per Second")
                ax2.set_title("Throughput vs Concurrency")
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # גרף 2: התפלגות ציוני איכות
        if "algorithm_quality" in report and isinstance(report["algorithm_quality"], dict):
            quality_data = report["algorithm_quality"]
            if "score_distribution" in quality_data:
                dist = quality_data["score_distribution"]
                
                categories = list(dist.keys())
                values = list(dist.values())
                colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(categories, values, color=colors)
                plt.xlabel("Score Categories")
                plt.ylabel("Percentage (%)")
                plt.title("Distribution of Match Quality Scores")
                
                # הוספת ערכים על העמודות
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{value:.1f}%', ha='center', va='bottom')
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/quality_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"גרפי ביצועים נשמרו ב: {output_dir}")

def main():
    """הפעלת בדיקות ביצועים"""
    tester = PerformanceTester()
    
    # יצירת דוח ביצועים
    report = tester.generate_performance_report()
    
    # יצירת גרפים
    tester.create_performance_visualizations(report)
    
    # הדפסת סיכום
    print("\n" + "="*50)
    print("סיכום בדיקות ביצועים")
    print("="*50)
    
    if "load_performance" in report:
        print("\nביצועי עומס:")
        for key, value in report["load_performance"].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                print(f"    Success Rate: {value.get('success_rate', 0):.1f}%")
                print(f"    Avg Response Time: {value.get('avg_response_time_ms', 0):.0f}ms")
                print(f"    Throughput: {value.get('throughput_requests_per_second', 0):.2f} req/sec")
    
    if "algorithm_quality" in report:
        quality = report["algorithm_quality"]
        print(f"\nאיכות אלגוריתם:")
        print(f"  דוגמאות נבדקו: {quality.get('total_tested', 0)}")
        print(f"  ציון ממוצע: {quality.get('avg_score_mean', 0):.3f}")
        print(f"  ציון מקסימלי ממוצע: {quality.get('max_score_mean', 0):.3f}")
        print(f"  התאמות ממוצעות למועמד: {quality.get('avg_matches_found', 0):.1f}")
        
        if "score_distribution" in quality:
            dist = quality["score_distribution"]
            print(f"  התפלגות איכות:")
            print(f"    מעולה (80%+): {dist.get('excellent', 0):.1f}%")
            print(f"    טוב (60-80%): {dist.get('good', 0):.1f}%")
            print(f"    בינוני (40-60%): {dist.get('fair', 0):.1f}%")
            print(f"    חלש (<40%): {dist.get('poor', 0):.1f}%")

if __name__ == "__main__":
    main()
